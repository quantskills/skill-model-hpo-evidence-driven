"""Core evidence-driven hyperparameter search runner."""

from __future__ import annotations

import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config_utils import ConfigError, json_default, load_config, optional_value, write_json, write_resolved_config
from data_adapter import build_holdout_test_window, build_panel, build_windows
from decision_provider import (
    ExternalDecisionRequired,
    decide_space_with_provider,
    request_final_selection_with_provider,
    resolve_decision_provider_config,
)
from evidence_builder import build_trial_evidence
from fast_evaluator import score_predictions
from grid_builder import resolve_grid_trials as build_grid_trials
from holdout_evaluator import evaluate_holdout
from llm_controller import llm_enabled
from llm_space_decider import build_decision_memory_item
from model_registry import ModelSpec
from final_selector import (
    build_final_selection_evidence,
    build_neighbor_plan,
    fallback_final_selection,
    resolve_final_selector_config,
    select_center_candidates,
    validate_final_selection,
)
from search_space import resolve_model_type, resolve_search_space, sample_params_with_metadata
from trainer import train_and_predict


DEFAULT_VALIDATION = {
    "method": "walk_forward",
    "window_unit": "trading_days",
    "train_window_days": 40,
    "valid_window_days": 10,
    "step_days": 10,
    "embargo_days": 6,
    "min_assets_per_date": 20,
}

DEFAULT_PREPROCESS = {
    "winsorize": {"enabled": True, "lower": 0.01, "upper": 0.99, "by": "date"},
    "normalize": {"method": "zscore_by_date"},
    "fillna": {"method": "cross_sectional_median"},
    "after_normalize_fillna": "zero",
}

FAST_SCORE_WEIGHT_PRESETS = {
    "balanced": {
        "rankic_ir": 0.40,
        "top_bottom_spread": 0.25,
        "positive_window_ratio": 0.15,
        "turnover_proxy": 0.10,
        "complexity_penalty": 0.05,
        "overfit_penalty": 0.05,
    },
    "robust": {
        "rankic_ir": 0.42,
        "top_bottom_spread": 0.15,
        "positive_window_ratio": 0.25,
        "turnover_proxy": 0.06,
        "complexity_penalty": 0.04,
        "overfit_penalty": 0.08,
    },
    "return_focus": {
        "rankic_ir": 0.30,
        "top_bottom_spread": 0.40,
        "positive_window_ratio": 0.12,
        "turnover_proxy": 0.08,
        "complexity_penalty": 0.04,
        "overfit_penalty": 0.06,
    },
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _git_commit(cwd: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(cwd), "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "unknown"


def _make_run_id(task_name: str) -> str:
    return f"{task_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}"


def _append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def _resolve_weights(cfg: Mapping[str, Any]) -> dict[str, float]:
    goal_cfg = _mapping(cfg.get("goal"))
    evaluation_cfg = _mapping(cfg.get("evaluation"))
    preference = str(goal_cfg.get("preference", "robust")).strip().lower()
    if preference not in FAST_SCORE_WEIGHT_PRESETS:
        allowed = ", ".join(sorted(FAST_SCORE_WEIGHT_PRESETS))
        raise ConfigError(f"Unsupported goal.preference={preference!r}; expected one of: {allowed}")
    weights = dict(FAST_SCORE_WEIGHT_PRESETS[preference])
    overrides = evaluation_cfg.get("fast_score_weights") or goal_cfg.get("fast_score_weights") or {}
    if overrides:
        if not isinstance(overrides, Mapping):
            raise ConfigError("fast_score_weights override must be a mapping")
        for key, value in overrides.items():
            if key not in weights:
                raise ConfigError(f"Unsupported fast_score weight override: {key}")
            weights[str(key)] = float(value)
    return weights


def _resolve_core_config(raw_cfg: Mapping[str, Any]) -> dict[str, Any]:
    cfg = dict(raw_cfg)
    input_cfg = _mapping(raw_cfg.get("input"))
    data_cfg = dict(_mapping(raw_cfg.get("data")))
    if "feature_path" not in data_cfg and input_cfg.get("feature_path"):
        data_cfg["feature_path"] = input_cfg.get("feature_path")
    if "label_path" not in data_cfg and input_cfg.get("label_path"):
        data_cfg["label_path"] = input_cfg.get("label_path")
    if "market_path" not in data_cfg and input_cfg.get("market_path"):
        data_cfg["market_path"] = input_cfg.get("market_path")
    data_cfg.setdefault("date_col", "date")
    data_cfg.setdefault("ticker_col", "ticker")
    data_cfg.setdefault("label_col", "y")
    data_cfg.setdefault("ticker_dtype", "preserve")
    data_cfg.setdefault("read_chunksize", 500_000)
    data_cfg.setdefault("compute_hash", False)
    data_cfg.setdefault("all_null_feature_policy", "allow")
    cfg["data"] = data_cfg

    task_cfg = dict(_mapping(raw_cfg.get("task")))
    task_cfg.setdefault("name", "evidence_adaptive_model_search")
    task_cfg.setdefault("mode", "hyperparameter_search")
    task_cfg.setdefault("seed", 42)
    cfg["task"] = task_cfg

    model_cfg = _mapping(raw_cfg.get("model"))
    search_cfg = dict(_mapping(raw_cfg.get("search")))
    model_type = str(search_cfg.get("model_type") or model_cfg.get("type") or raw_cfg.get("model_type") or "lgbm").strip().lower()
    if model_type == "auto":
        model_type = "lgbm"
    search_cfg["model_type"] = model_type
    max_trials = int(search_cfg.get("max_trials", 20))
    search_cfg.setdefault("method", "adaptive_tpe")
    search_cfg.setdefault("max_trials", max_trials)
    search_cfg.setdefault("max_rounds", 2)
    search_cfg.setdefault("trials_per_round", max_trials)
    search_cfg.setdefault("random_start_trials", min(8, max_trials))
    search_cfg.setdefault("top_fraction", 0.30)
    search_cfg.setdefault("seed", int(task_cfg.get("seed", 42)))
    search_cfg.setdefault("normalize_method", None)
    search_cfg.setdefault("allow_overlapping_validation", False)
    search_cfg.setdefault("fixed_params", {})
    cfg["search"] = search_cfg

    validation_cfg = dict(DEFAULT_VALIDATION)
    validation_cfg.update(dict(_mapping(raw_cfg.get("validation"))))
    cfg["validation"] = validation_cfg

    training_cfg = {"label_window": 5}
    training_cfg.update(dict(_mapping(raw_cfg.get("training"))))
    cfg["training"] = training_cfg

    time_cfg = {"signal_date_policy": "feature_date", "trade_lag_days": 1, "locked_test_start": None}
    time_cfg.update(dict(_mapping(raw_cfg.get("time"))))
    cfg["time"] = time_cfg

    evaluation_cfg = dict(_mapping(raw_cfg.get("evaluation")))
    evaluation_cfg.setdefault("inner_loop", "fast_evaluator")
    evaluation_cfg.setdefault("objective", "rankic_ir")
    evaluation_cfg["fast_score_weights"] = _resolve_weights(cfg)
    cfg["evaluation"] = evaluation_cfg
    cfg["fast_evaluator"] = dict(_mapping(raw_cfg.get("fast_evaluator")) or {"top_quantile": 0.2, "bottom_quantile": 0.2})
    cfg["preprocess"] = dict(_mapping(raw_cfg.get("preprocess")) or DEFAULT_PREPROCESS)
    cfg["space_controller"] = dict(_mapping(raw_cfg.get("space_controller")) or {"enabled": True, "mode": "rule"})
    cfg["llm"] = dict(_mapping(raw_cfg.get("llm")) or {"enabled": False})
    cfg["decision_provider"] = dict(_mapping(raw_cfg.get("decision_provider")) or {})
    cfg["final_selector"] = resolve_final_selector_config(
        _mapping(raw_cfg.get("final_selector")),
        model_type=model_type,
    )
    return cfg


def _check_validation_overlap(cfg: Mapping[str, Any]) -> None:
    allow = bool(optional_value(cfg, ["search", "allow_overlapping_validation"], False))
    if allow:
        return
    valid_n = int(optional_value(cfg, ["validation", "valid_window_days"], 0))
    step_n = int(optional_value(cfg, ["validation", "step_days"], 0))
    if valid_n and step_n and step_n < valid_n:
        raise ConfigError("search requires validation.step_days >= validation.valid_window_days unless search.allow_overlapping_validation=true")


def _status_row_base(trial_id: str, trial_index: int, model_type: str, params: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "trial_id": trial_id,
        "trial_index": trial_index,
        "model_type": model_type,
        "params": dict(params),
        "status": "started",
    }


def _flatten_trial(row: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(row)
    params = out.pop("params", {})
    sample_meta = out.pop("sample_meta", {})
    out["params_json"] = json.dumps(params, ensure_ascii=False, sort_keys=True, default=json_default)
    out["sample_meta_json"] = json.dumps(sample_meta, ensure_ascii=False, sort_keys=True, default=json_default)
    return out


def _normalize_search_method(method: Any) -> str:
    raw = str(method or "evidence_driven").strip().lower()
    aliases = {
        "adaptive": "evidence_driven",
        "adaptive_tpe": "evidence_driven",
        "tpe": "evidence_driven",
        "tpe_like": "evidence_driven",
        "evidence": "evidence_driven",
        "evidence_driven": "evidence_driven",
        "evidence-driven": "evidence_driven",
        "fixed_grid": "grid",
        "grid_search": "grid",
        "grid-search": "grid",
        "grid": "grid",
    }
    normalized = aliases.get(raw, raw)
    if normalized not in {"evidence_driven", "grid"}:
        raise ConfigError("search.method must be one of: evidence_driven, grid")
    return normalized


def _resolve_grid_trials(
    search_cfg: Mapping[str, Any],
    search_space: Mapping[str, Mapping[str, Any]],
    model_type: str,
    max_trials: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return build_grid_trials(
        search_cfg=search_cfg,
        search_space=search_space,
        model_type=model_type,
        max_trials=max_trials,
    )


def _boundary_params(ok: pd.DataFrame, space: Mapping[str, Any]) -> list[str]:
    top = ok.sort_values("score", ascending=False).head(max(3, int(math.ceil(len(ok) * 0.3))))
    params: list[str] = []
    for name, spec in space.items():
        kind = spec.get("type")
        values = []
        for raw in top.get("params_json", []):
            try:
                values.append(json.loads(raw).get(name))
            except Exception:
                continue
        if not values:
            continue
        if kind == "choice":
            allowed = list(spec.get("values", []))
            if allowed and (all(v == allowed[0] for v in values) or all(v == allowed[-1] for v in values)):
                params.append(name)
        elif kind in {"uniform", "loguniform", "quniform", "qloguniform"}:
            low, high = float(spec["low"]), float(spec["high"])
            numeric = np.array([float(v) for v in values if v is not None], dtype="float64")
            if numeric.size and (np.nanmean(numeric) <= low + 0.1 * (high - low) or np.nanmean(numeric) >= high - 0.1 * (high - low)):
                params.append(name)
    return params


def _diagnose_failures(leaderboard: pd.DataFrame, space: Mapping[str, Any]) -> dict[str, Any]:
    if leaderboard.empty or (leaderboard["status"] == "ok").sum() == 0:
        return {"failure_modes": ["no_valid_trials"]}
    ok = leaderboard[leaderboard["status"] == "ok"].copy()
    best = ok.sort_values("score", ascending=False).iloc[0]
    modes: list[str] = []
    objective = str(best.get("objective") or "").lower()
    if objective == "rmse":
        train_rmse = float(best.get("window_train_rmse", np.nan))
        valid_rmse = float(best.get("window_valid_rmse", np.nan))
        if np.isfinite(train_rmse) and np.isfinite(valid_rmse) and train_rmse > 0 and valid_rmse / train_rmse > 1.5:
            modes.append("loss_overfit_gap_high")
        if len(ok) >= 3 and "valid_rmse" in ok:
            valid_rmse_values = pd.to_numeric(ok["valid_rmse"], errors="coerce").dropna()
            if len(valid_rmse_values) >= 3 and float(valid_rmse_values.std(ddof=1)) < 1e-9:
                modes.append("flat_rmse_response")
    else:
        if float(best.get("overfit_penalty", 0.0)) > 0.35:
            modes.append("overfit_high")
        if float(best.get("positive_window_ratio", 0.0)) < 0.5:
            modes.append("unstable_across_windows")
        if float(best.get("turnover_proxy", 0.0)) > 0.85:
            modes.append("turnover_too_high")
        if not np.isfinite(float(best.get("mean_rankic", np.nan))) or abs(float(best.get("mean_rankic", 0.0))) < 1e-6:
            modes.append("no_signal")
        if len(ok) >= 3 and float(ok["score"].std(ddof=1)) < 1e-6:
            modes.append("flat_search_response")
    boundary_params = _boundary_params(ok, space)
    if boundary_params:
        modes.append("boundary_optimum")
    return {
        "failure_modes": modes or ["none"],
        "best_trial_id": best.get("trial_id"),
        "best_score": float(best.get("score", np.nan)),
        "boundary_params": boundary_params,
        "num_ok_trials": int(len(ok)),
    }


def _leaderboard_from_history(trial_history: list[dict[str, Any]]) -> pd.DataFrame:
    leaderboard = pd.DataFrame([_flatten_trial(row) for row in trial_history])
    if not leaderboard.empty:
        leaderboard["status_rank"] = (leaderboard["status"] != "ok").astype(int)
        leaderboard = leaderboard.sort_values(["status_rank", "score"], ascending=[True, False]).drop(columns=["status_rank"]).reset_index(drop=True)
    return leaderboard


def _render_search_report(path: Path, summary: Mapping[str, Any], leaderboard: pd.DataFrame, diagnostics: Mapping[str, Any]) -> None:
    lines = ["# Model Hyperparameter Search Report", ""]
    lines.append("## Run Metadata")
    for key in [
        "run_id",
        "task_name",
        "model_type",
        "search_method",
        "sampler",
        "probe_fraction",
        "evaluation_objective",
        "max_trials",
        "grid_enabled",
        "num_grid_trials",
        "best_trial_id",
        "best_score",
    ]:
        lines.append(f"- `{key}`: {summary.get(key)}")
    lines.append("")
    if summary.get("grid_enabled"):
        grid_manifest = summary.get("grid_manifest") or {}
        lines.append("## Grid Search")
        for key in ["strategy", "source", "num_candidates_generated", "num_trials_selected", "selection", "seed", "truncated"]:
            lines.append(f"- `{key}`: {grid_manifest.get(key)}")
        lines.append("")
    lines.append("## Diagnostics")
    for item in diagnostics.get("failure_modes", []):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Top Trials")
    ok = leaderboard[leaderboard["status"] == "ok"].sort_values("score", ascending=False).head(10)
    cols = [c for c in ["trial_id", "sampler", "probe_type", "score", "loss", "valid_rmse", "valid_mae", "valid_r2", "fast_score", "mean_rankic", "rankic_ir", "top_bottom_spread", "positive_window_ratio", "turnover_proxy", "overfit_penalty", "complexity_penalty"] if c in ok]
    if ok.empty or not cols:
        lines.append("No successful trials.")
    else:
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for _, row in ok[cols].iterrows():
            values = []
            for col in cols:
                value = row[col]
                values.append(f"{value:.6g}" if isinstance(value, float) else str(value).replace("|", "\\|"))
            lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    lines.append("## Search Space Adaptation")
    decisions = summary.get("space_controller_decisions") or []
    lines.append(f"- space_controller_enabled: {summary.get('space_controller_enabled')}")
    lines.append(f"- space_controller_mode: {summary.get('space_controller_mode')}")
    lines.append(f"- num_space_versions: {summary.get('num_space_versions')}")
    if decisions:
        lines.append("| round | action | reason | changed_params |")
        lines.append("| ---: | --- | --- | --- |")
        for record in decisions:
            decision = record.get("validated_decision", record)
            changed = ", ".join(str(item.get("param")) for item in decision.get("changed_params", []))
            source = record.get("source", "controller")
            lines.append(f"| {decision.get('round_id', record.get('round_id'))} | {source}:{decision.get('action')} | {decision.get('reason')} | {changed} |")
    elif summary.get("grid_enabled"):
        lines.append("- space controller is disabled for deterministic grid search")
    else:
        lines.append("- no controller decision was recorded")
    lines.append("")
    lines.append("Search trials are scored by validation metrics only; holdout test metrics are reported after hyperparameter selection and are not fed back into the controller.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_one_trial(
    *,
    trial_id: str,
    trial_index: int,
    model_type: str,
    params: Mapping[str, Any],
    panel_data: Any,
    windows: list[Any],
    cfg: Mapping[str, Any],
    seed: int,
    normalize_method: Any,
) -> tuple[dict[str, Any], pd.DataFrame | None, pd.DataFrame | None]:
    row = _status_row_base(trial_id, trial_index, model_type, params)
    try:
        spec = ModelSpec(name=trial_id, type="lightgbm" if model_type == "lgbm" else model_type, params=params)
        predictions, window_metrics = train_and_predict(
            panel_data.panel,
            panel_data.feature_columns,
            windows,
            [spec],
            cfg,
            seed=seed + trial_index * 997,
            normalize_method=str(normalize_method) if normalize_method else None,
        )
        duplicates = int(predictions.duplicated(["model_name", "date", "ticker"]).sum())
        if duplicates:
            raise ValueError(f"Trial produced duplicated (model_name,date,ticker) predictions: {duplicates}")
        best_summary, _ = score_predictions(predictions, cfg)
        score = float(best_summary.get("objective_score", np.nan))
        if not np.isfinite(score):
            raise ValueError("Trial produced non-finite objective score")
        row.update({
            "status": "ok",
            "score": score,
            "loss": best_summary.get("valid_rmse") if str(best_summary.get("objective")) == "rmse" else -score,
            "objective": best_summary.get("objective"),
            "fast_score": best_summary.get("fast_score"),
            "valid_rmse": best_summary.get("valid_rmse"),
            "valid_mae": best_summary.get("valid_mae"),
            "valid_r2": best_summary.get("valid_r2"),
            "window_train_rmse": best_summary.get("window_train_rmse"),
            "window_valid_rmse": best_summary.get("window_valid_rmse"),
            "window_train_r2": best_summary.get("window_train_r2"),
            "window_valid_r2": best_summary.get("window_valid_r2"),
            "mean_ic": best_summary.get("mean_ic"),
            "mean_rankic": best_summary.get("mean_rankic"),
            "icir": best_summary.get("icir"),
            "rankic_ir": best_summary.get("rankic_ir"),
            "top_bottom_spread": best_summary.get("top_bottom_spread"),
            "positive_window_ratio": best_summary.get("positive_window_ratio"),
            "turnover_proxy": best_summary.get("turnover_proxy"),
            "instability_penalty": best_summary.get("instability_penalty"),
            "complexity_penalty": best_summary.get("complexity_penalty"),
            "overfit_penalty": best_summary.get("overfit_penalty"),
            "num_prediction_rows": int(best_summary.get("num_prediction_rows", len(predictions))),
        })
        window_metrics = window_metrics.copy()
        window_metrics.insert(0, "trial_id", trial_id)
        window_metrics["trial_score"] = score
        return row, predictions, window_metrics
    except Exception as exc:
        row.update({"status": "failed", "score": np.nan, "loss": np.inf, "error": f"{type(exc).__name__}: {exc}"})
        return row, None, None


def run_search(config_path: Path, output_root: Path) -> dict[str, Any]:
    raw_cfg = load_config(config_path)
    cfg = _resolve_core_config(raw_cfg)
    _check_validation_overlap(cfg)

    task_name = str(optional_value(cfg, ["task", "name"], "evidence_adaptive_model_search"))
    run_id = str(optional_value(cfg, ["task", "run_id"], None) or _make_run_id(task_name))
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_type = resolve_model_type(cfg)
    space = resolve_search_space(cfg, model_type)
    search_cfg = dict(_mapping(cfg.get("search")))
    method = _normalize_search_method(search_cfg.get("method", "evidence_driven"))
    max_trials = int(search_cfg.get("max_trials", 20))
    random_start_trials = int(search_cfg.get("random_start_trials", min(8, max_trials)))
    top_fraction = float(search_cfg.get("top_fraction", 0.30))
    sampler = str(search_cfg.get("sampler", "adaptive")).strip().lower()
    if sampler == "fixed_grid":
        sampler = "grid"
    if sampler == "grid":
        method = "grid"
    if method == "grid":
        sampler = "grid"
    sampling_method = "adaptive_tpe" if method == "evidence_driven" else method
    probe_fraction = float(search_cfg.get("probe_fraction", 0.0))
    normalize_method = search_cfg.get("normalize_method")
    seed = int(search_cfg.get("seed", optional_value(cfg, ["task", "seed"], 42)))
    max_rounds = int(search_cfg.get("max_rounds", 1))
    trials_per_round = int(search_cfg.get("trials_per_round", max_trials))
    if max_trials <= 0:
        raise ConfigError("search.max_trials must be positive")
    if random_start_trials < 0:
        raise ConfigError("search.random_start_trials must be non-negative")
    if max_rounds <= 0 or trials_per_round <= 0:
        raise ConfigError("search.max_rounds and search.trials_per_round must be positive")
    if sampler not in {"adaptive", "evidence_probe", "structured_probe", "local_probe", "grid"}:
        raise ConfigError("search.sampler must be one of: adaptive, evidence_probe, structured_probe, local_probe, grid")
    if not 0.0 <= probe_fraction <= 1.0:
        raise ConfigError("search.probe_fraction must be in [0, 1]")
    use_grid = method == "grid"
    if use_grid:
        grid_trials, grid_manifest = _resolve_grid_trials(search_cfg, space, model_type, max_trials)
        if not grid_trials:
            raise ConfigError("grid search generated no grid trials")
        if max_trials > len(grid_trials):
            max_trials = len(grid_trials)
        trials_per_round = min(trials_per_round, max_trials)
    else:
        grid_trials = []
        grid_manifest = {"enabled": False}
    search_cfg["method"] = method
    search_cfg["sampler"] = sampler
    search_cfg["max_trials"] = max_trials
    cfg["search"] = search_cfg

    panel_data = build_panel(cfg)
    windows = build_windows(panel_data.panel, cfg)
    holdout_window = build_holdout_test_window(panel_data.panel, cfg)
    rng = np.random.default_rng(seed)
    use_llm = llm_enabled(cfg)
    controller_cfg = dict(_mapping(cfg.get("space_controller")))
    controller_cfg.setdefault("trials_per_round", trials_per_round)
    controller_cfg.setdefault("max_next_round_trials", trials_per_round)
    use_space_controller = bool(controller_cfg.get("enabled", True))
    if use_grid and use_space_controller:
        use_space_controller = False
    controller_mode = str(controller_cfg.get("mode", "rule")).lower()
    if use_grid:
        controller_mode = "disabled_for_grid"
    evidence_boundary_margin = float(controller_cfg.get("boundary_margin", 0.15))
    decision_provider_config = resolve_decision_provider_config(cfg, run_dir=run_dir)
    cfg["decision_provider"] = decision_provider_config

    manifest = {
        "run_id": run_id,
        "task_name": task_name,
        "commit": _git_commit(Path(__file__).resolve().parent),
        "model_type": model_type,
        "search_method": method,
        "sampler": sampler,
        "probe_fraction": probe_fraction,
        "evaluation_objective": optional_value(cfg, ["evaluation", "objective"], "fast_score"),
        "actual_sampler": (
            "grid"
            if use_grid
            else (
                f"{sampler}_over_internal_adaptive_tpe_style"
                if sampler in {"evidence_probe", "structured_probe", "local_probe"}
                else ("internal_adaptive_tpe_style" if sampling_method in {"adaptive_tpe", "tpe", "tpe_like"} else "random")
            )
        ),
        "grid_enabled": use_grid,
        "grid_manifest": grid_manifest,
        "num_grid_trials": len(grid_trials),
        "max_trials": max_trials,
        "max_rounds": max_rounds,
        "trials_per_round": trials_per_round,
        "random_start_trials": random_start_trials,
        "top_fraction": top_fraction,
        "llm_enabled": use_llm,
        "decision_provider_type": decision_provider_config.get("type"),
        "decision_provider": dict(decision_provider_config),
        "space_controller_enabled": use_space_controller,
        "space_controller_mode": controller_mode,
        "space_controller": dict(controller_cfg),
        "seed": seed,
        "data_metadata": panel_data.metadata,
        "num_features": len(panel_data.feature_columns),
        "feature_columns": panel_data.feature_columns,
        "num_windows": len(windows),
        "holdout_enabled": holdout_window is not None,
        "holdout_window": holdout_window.to_dict() if holdout_window is not None else None,
        "warnings": panel_data.warnings,
    }
    write_json(manifest, run_dir / "search_manifest.json")
    if use_grid:
        write_json(grid_trials, run_dir / "grid_trials_resolved.json")
        write_json(grid_manifest, run_dir / "grid_manifest.json")
    base_space = json.loads(json.dumps(space, ensure_ascii=False, default=json_default))
    space_versions: list[dict[str, Any]] = [{"version": 0, "round_id": 0, "reason": "initial_space", "space": space}]
    write_json(space_versions, run_dir / "search_space_versions.json")
    write_resolved_config(cfg, run_dir / "resolved_config.yaml")

    trial_history: list[dict[str, Any]] = []
    window_metric_frames: list[pd.DataFrame] = []
    round_history: list[dict[str, Any]] = []
    trial_evidence_history: list[dict[str, Any]] = []
    space_controller_decisions: list[dict[str, Any]] = []
    decision_memory: list[dict[str, Any]] = []
    pending_memory: dict[str, Any] | None = None
    best_score = -np.inf
    best_trial: dict[str, Any] | None = None
    trials_path = run_dir / "trials.jsonl"

    trial_index = 0
    stop_requested = False
    for round_id in range(max_rounds):
        if trial_index >= max_trials or stop_requested:
            break
        current_round_trials = min(trials_per_round, max_trials - trial_index)
        round_start = trial_index
        for trial_in_round in range(current_round_trials):
            if use_grid:
                params = dict(grid_trials[trial_index])
                sample_meta = {
                    "sampler": "grid",
                    "probe_applied": False,
                    "probe_type": "grid",
                    "grid_index": trial_index,
                    "grid_strategy": grid_manifest.get("strategy"),
                    "trial_in_round": trial_in_round,
                    "trials_per_round": current_round_trials,
                }
            else:
                params, sample_meta = sample_params_with_metadata(
                    space,
                    rng,
                    trial_history,
                    model_type=model_type,
                    method=sampling_method,
                    random_start_trials=random_start_trials,
                    top_fraction=top_fraction,
                    sampler=sampler,
                    probe_fraction=probe_fraction,
                    trial_in_round=trial_in_round,
                    trials_per_round=current_round_trials,
                )
            trial_id = f"{model_type}_trial_{trial_index:04d}"
            row, _, window_metrics = _run_one_trial(
                trial_id=trial_id,
                trial_index=trial_index,
                model_type=model_type,
                params=params,
                panel_data=panel_data,
                windows=windows,
                cfg=cfg,
                seed=seed,
                normalize_method=normalize_method,
            )
            row["round_id"] = round_id
            row["sampler"] = sample_meta.get("sampler")
            row["probe_type"] = sample_meta.get("probe_type")
            row["probe_applied"] = sample_meta.get("probe_applied")
            row["sample_meta"] = sample_meta
            trial_history.append(dict(row))
            _append_jsonl(trials_path, row)
            if window_metrics is not None:
                window_metrics.insert(1, "round_id", round_id)
                window_metric_frames.append(window_metrics)
            if row.get("status") == "ok" and float(row.get("score", -np.inf)) > best_score:
                best_score = float(row["score"])
                best_trial = dict(row)
            trial_index += 1

        leaderboard_snapshot = _leaderboard_from_history(trial_history)
        round_diagnostics = _diagnose_failures(leaderboard_snapshot, space)
        round_entry: dict[str, Any] = {
            "round_id": round_id,
            "method": method,
            "space_version": len(space_versions) - 1,
            "trial_start": round_start,
            "trial_end": trial_index - 1,
            "num_trials": current_round_trials,
            "diagnostics": round_diagnostics,
        }
        trial_evidence = build_trial_evidence(
            round_id=round_id,
            model_type=model_type,
            current_space=space,
            leaderboard=leaderboard_snapshot,
            diagnostics=round_diagnostics,
            top_fraction=top_fraction,
            boundary_margin=evidence_boundary_margin,
        )
        evidence_path = run_dir / f"trial_evidence_round_{round_id}.json"
        write_json(trial_evidence, evidence_path)
        trial_evidence_history.append(trial_evidence)
        round_entry["trial_evidence_path"] = evidence_path.name
        if pending_memory is not None:
            pending_memory["memory_item"] = build_decision_memory_item(
                round_id=int(pending_memory["round_id"]),
                decision_record=pending_memory["decision_record"],
                evidence_before=pending_memory["evidence_before"],
                evidence_after=trial_evidence,
            )
            decision_memory.append(pending_memory["memory_item"])
            pending_memory = None

        if use_space_controller and round_id < max_rounds - 1 and trial_index < max_trials:
            decision_path = run_dir / f"space_decision_round_{round_id}.json"
            try:
                decision_record = decide_space_with_provider(
                    cfg=cfg,
                    provider_config=decision_provider_config,
                    trial_evidence=trial_evidence,
                    current_space=space,
                    base_space=base_space,
                    controller_config=controller_cfg,
                    model_type=model_type,
                    decision_memory=decision_memory,
                )
            except ExternalDecisionRequired as exc:
                requirement = exc.to_dict()
                write_json(requirement, run_dir / "external_decision_required.json")
                round_entry["external_decision_required"] = requirement
                round_history.append(round_entry)
                write_json(space_versions, run_dir / "search_space_versions.json")
                write_json(round_history, run_dir / "round_history.json")
                write_json(trial_evidence_history, run_dir / "trial_evidence_history.json")
                write_json(space_controller_decisions, run_dir / "space_controller_decisions.json")
                write_json(decision_memory, run_dir / "decision_memory.json")
                leaderboard = _leaderboard_from_history(trial_history)
                leaderboard.to_csv(run_dir / "trial_leaderboard.csv", index=False)
                if window_metric_frames:
                    pd.concat(window_metric_frames, ignore_index=True).to_csv(run_dir / "trial_window_metrics.csv", index=False)
                else:
                    pd.DataFrame().to_csv(run_dir / "trial_window_metrics.csv", index=False)
                diagnostics = _diagnose_failures(leaderboard, space)
                write_json(diagnostics, run_dir / "failure_modes.json")
                num_trials_run = int(len(trial_history))
                num_successful_trials = int((leaderboard["status"] == "ok").sum()) if not leaderboard.empty else 0
                summary = {
                    "run_id": run_id,
                    "task_name": task_name,
                    "model_type": model_type,
                    "status": "external_decision_required",
                    "external_decision_required": requirement,
                    "search_method": method,
                    "sampler": sampler,
                    "probe_fraction": probe_fraction,
                    "evaluation_objective": optional_value(cfg, ["evaluation", "objective"], "fast_score"),
                    "max_trials": max_trials,
                    "grid_enabled": use_grid,
                    "grid_manifest": grid_manifest,
                    "num_grid_trials": len(grid_trials),
                    "num_trials_run": num_trials_run,
                    "num_successful_trials": num_successful_trials,
                    "num_failed_trials": int(num_trials_run - num_successful_trials),
                    "num_rounds": int(len(round_history)),
                    "llm_enabled": use_llm,
                    "decision_provider_type": decision_provider_config.get("type"),
                    "decision_provider": dict(decision_provider_config),
                    "space_controller_enabled": use_space_controller,
                    "space_controller_mode": controller_mode,
                    "space_controller_decisions": space_controller_decisions,
                    "decision_memory": decision_memory,
                    "num_space_versions": int(len(space_versions)),
                    "final_selector_enabled": bool(_mapping(cfg.get("final_selector")).get("enabled")),
                    "diagnostics": diagnostics,
                    "data_metadata": panel_data.metadata,
                    "warnings": panel_data.warnings,
                }
                write_json(summary, run_dir / "run_summary.json")
                _render_search_report(run_dir / "search_report.md", summary, leaderboard, diagnostics)
                return summary
            decision = dict(decision_record.get("validated_decision") or {})
            compact_record = {key: value for key, value in decision_record.items() if key != "llm_evidence"}
            if decision_record.get("llm_evidence") is not None:
                write_json(decision_record["llm_evidence"], run_dir / f"llm_guarded_evidence_round_{round_id}.json")
            space_controller_decisions.append(compact_record)
            round_entry["space_controller_decision"] = compact_record
            round_entry["space_controller_decision_path"] = decision_path.name
            if decision.get("action") == "stop":
                stop_requested = True
            elif decision.get("next_search_space") and decision.get("action") != "keep":
                space = dict(decision["next_search_space"])
                space_versions.append({
                    "version": len(space_versions),
                    "round_id": round_id + 1,
                    "reason": f"{decision_record.get('source')}_{decision.get('action', 'update')}",
                    "controller_reason": decision.get("reason"),
                    "changed_params": decision.get("changed_params", []),
                    "space": space,
                })
            pending_memory = {"round_id": round_id, "decision_record": compact_record, "evidence_before": trial_evidence}
            if decision.get("next_round_trials"):
                raw_next_round_trials = int(decision["next_round_trials"])
                applied_next_round_trials = max(1, min(raw_next_round_trials, int(search_cfg.get("trials_per_round", trials_per_round))))
                decision["next_round_trials_raw"] = raw_next_round_trials
                decision["next_round_trials_applied"] = applied_next_round_trials
                decision["next_round_trials_clamped"] = raw_next_round_trials != applied_next_round_trials
                if isinstance(round_entry.get("space_controller_decision"), dict):
                    round_entry["space_controller_decision"]["validated_decision"] = decision
                if space_controller_decisions:
                    space_controller_decisions[-1]["validated_decision"] = decision
                trials_per_round = applied_next_round_trials
            write_json(compact_record, decision_path)
        round_history.append(round_entry)
        write_json(space_versions, run_dir / "search_space_versions.json")
        write_json(round_history, run_dir / "round_history.json")
        write_json(trial_evidence_history, run_dir / "trial_evidence_history.json")
        write_json(space_controller_decisions, run_dir / "space_controller_decisions.json")
        write_json(decision_memory, run_dir / "decision_memory.json")

    leaderboard = _leaderboard_from_history(trial_history)
    leaderboard.to_csv(run_dir / "trial_leaderboard.csv", index=False)
    if window_metric_frames:
        pd.concat(window_metric_frames, ignore_index=True).to_csv(run_dir / "trial_window_metrics.csv", index=False)
    else:
        pd.DataFrame().to_csv(run_dir / "trial_window_metrics.csv", index=False)

    diagnostics = _diagnose_failures(leaderboard, space)
    write_json(diagnostics, run_dir / "failure_modes.json")
    write_json(round_history, run_dir / "round_history.json")
    write_json(space_versions, run_dir / "search_space_versions.json")
    write_json(trial_evidence_history, run_dir / "trial_evidence_history.json")
    if pending_memory is not None:
        pending_memory["memory_item"] = build_decision_memory_item(
            round_id=int(pending_memory["round_id"]),
            decision_record=pending_memory["decision_record"],
            evidence_before=pending_memory["evidence_before"],
            evidence_after=None,
        )
        decision_memory.append(pending_memory["memory_item"])
        pending_memory = None
    write_json(space_controller_decisions, run_dir / "space_controller_decisions.json")
    write_json(decision_memory, run_dir / "decision_memory.json")

    num_trials_run = int(len(trial_history))
    num_successful_trials = int((leaderboard["status"] == "ok").sum()) if not leaderboard.empty else 0
    num_failed_trials = int(num_trials_run - num_successful_trials)

    if best_trial is None:
        summary = {
            "run_id": run_id,
            "task_name": task_name,
            "model_type": model_type,
            "status": "failed",
            "search_method": method,
            "sampler": sampler,
            "probe_fraction": probe_fraction,
            "evaluation_objective": optional_value(cfg, ["evaluation", "objective"], "fast_score"),
            "max_trials": max_trials,
            "grid_enabled": use_grid,
            "grid_manifest": grid_manifest,
            "num_grid_trials": len(grid_trials),
            "num_trials_run": num_trials_run,
            "num_successful_trials": num_successful_trials,
            "num_failed_trials": num_failed_trials,
            "num_rounds": int(len(round_history)),
            "llm_enabled": use_llm,
            "space_controller_enabled": use_space_controller,
            "space_controller_mode": controller_mode,
            "space_controller_decisions": space_controller_decisions,
            "decision_memory": decision_memory,
            "num_space_versions": int(len(space_versions)),
            "final_selector_enabled": bool(_mapping(cfg.get("final_selector")).get("enabled")),
            "diagnostics": diagnostics,
            "data_metadata": panel_data.metadata,
            "warnings": panel_data.warnings,
        }
        write_json(summary, run_dir / "run_summary.json")
        _render_search_report(run_dir / "search_report.md", summary, leaderboard, diagnostics)
        return summary

    score_best_trial = dict(best_trial)
    selected_trial = dict(best_trial)
    final_selector_cfg = dict(_mapping(cfg.get("final_selector")))
    final_selection: dict[str, Any] = {"enabled": False}
    final_neighbor_rows: list[dict[str, Any]] = []

    score_best_params = {
        "model_type": model_type,
        "trial_id": score_best_trial["trial_id"],
        "objective": score_best_trial.get("objective"),
        "score": score_best_trial["score"],
        "loss": score_best_trial["loss"],
        "valid_rmse": score_best_trial.get("valid_rmse"),
        "valid_mae": score_best_trial.get("valid_mae"),
        "valid_r2": score_best_trial.get("valid_r2"),
        "fast_score": score_best_trial.get("fast_score"),
        "params": score_best_trial["params"],
    }
    write_json(score_best_params, run_dir / "score_best_params.json")

    if final_selector_cfg.get("enabled"):
        center_candidates = select_center_candidates(trial_history, top_k=int(final_selector_cfg["top_k"]))
        final_rng = np.random.default_rng(seed + 8_888_881)
        neighbor_plan = build_neighbor_plan(
            center_candidates,
            search_space=space,
            model_type=model_type,
            selector_cfg=final_selector_cfg,
            rng=final_rng,
        )
        final_neighbor_window_metric_frames: list[pd.DataFrame] = []
        for final_index, item in enumerate(neighbor_plan):
            row, _, window_metrics = _run_one_trial(
                trial_id=str(item["neighbor_trial_id"]),
                trial_index=num_trials_run + final_index,
                model_type=model_type,
                params=item["params"],
                panel_data=panel_data,
                windows=windows,
                cfg=cfg,
                seed=seed + 8_000_003,
                normalize_method=normalize_method,
            )
            row["center_trial_id"] = item["center_trial_id"]
            row["center_rank"] = item["center_rank"]
            row["center_score"] = item["center_score"]
            row["neighbor_index"] = item["neighbor_index"]
            row["sampler"] = "final_selector"
            row["probe_type"] = "neighborhood"
            row["probe_applied"] = True
            row["sample_meta"] = {
                "sampler": "final_selector",
                "probe_type": "neighborhood",
                "center_trial_id": item["center_trial_id"],
                "center_rank": item["center_rank"],
                "neighbor_index": item["neighbor_index"],
            }
            final_neighbor_rows.append(row)
            if window_metrics is not None:
                window_metrics = window_metrics.copy()
                window_metrics.insert(1, "center_trial_id", item["center_trial_id"])
                window_metrics.insert(2, "neighbor_index", item["neighbor_index"])
                final_neighbor_window_metric_frames.append(window_metrics)

        final_neighbor_leaderboard = _leaderboard_from_history(final_neighbor_rows)
        final_neighbor_leaderboard.to_csv(run_dir / "final_neighbor_trials.csv", index=False)
        if final_neighbor_window_metric_frames:
            pd.concat(final_neighbor_window_metric_frames, ignore_index=True).to_csv(
                run_dir / "final_neighbor_window_metrics.csv",
                index=False,
            )
        else:
            pd.DataFrame().to_csv(run_dir / "final_neighbor_window_metrics.csv", index=False)

        final_selection_evidence = build_final_selection_evidence(
            model_type=model_type,
            objective=str(optional_value(cfg, ["evaluation", "objective"], "fast_score")),
            candidates=center_candidates,
            all_trials=trial_history,
            neighbor_rows=final_neighbor_rows,
            selector_cfg=final_selector_cfg,
        )
        write_json(final_selection_evidence, run_dir / "final_selection_evidence.json")
        try:
            raw_final_decision, final_source, final_error, final_external_paths = request_final_selection_with_provider(
                cfg=cfg,
                provider_config=decision_provider_config,
                evidence=final_selection_evidence,
            )
            if raw_final_decision is None:
                final_selection = fallback_final_selection(
                    evidence=final_selection_evidence,
                    candidates=center_candidates,
                    selector_cfg=final_selector_cfg,
                    source=final_source,
                    error=final_error or "final_selection_provider_returned_no_decision",
                )
            else:
                final_selection = validate_final_selection(
                    raw_final_decision,
                    evidence=final_selection_evidence,
                    candidates=center_candidates,
                    selector_cfg=final_selector_cfg,
                    source=final_source,
                )
            if final_external_paths:
                final_selection["external_paths"] = final_external_paths
        except ExternalDecisionRequired as exc:
            requirement = exc.to_dict()
            write_json(requirement, run_dir / "external_decision_required.json")
            final_selection = {
                "enabled": True,
                "status": "external_decision_required",
                "external_decision_required": requirement,
                "validated_selection": {},
                "config": dict(final_selector_cfg),
            }
            write_json(final_selection, run_dir / "final_selection.json")
            summary = {
                "run_id": run_id,
                "task_name": task_name,
                "model_type": model_type,
                "search_method": method,
                "sampler": sampler,
                "probe_fraction": probe_fraction,
                "evaluation_objective": optional_value(cfg, ["evaluation", "objective"], "fast_score"),
                "status": "external_decision_required",
                "external_decision_required": requirement,
                "max_trials": max_trials,
                "grid_enabled": use_grid,
                "grid_manifest": grid_manifest,
                "num_grid_trials": len(grid_trials),
                "num_trials_run": num_trials_run,
                "num_successful_trials": num_successful_trials,
                "num_failed_trials": num_failed_trials,
                "num_rounds": int(len(round_history)),
                "llm_enabled": use_llm,
                "decision_provider_type": decision_provider_config.get("type"),
                "decision_provider": dict(decision_provider_config),
                "space_controller_enabled": use_space_controller,
                "space_controller_mode": controller_mode,
                "space_controller_decisions": space_controller_decisions,
                "decision_memory": decision_memory,
                "num_space_versions": int(len(space_versions)),
                "score_best_trial_id": score_best_trial["trial_id"],
                "score_best_score": score_best_trial["score"],
                "score_best_params": score_best_trial["params"],
                "final_selector_enabled": bool(final_selector_cfg.get("enabled")),
                "num_final_neighbor_trials": int(len(final_neighbor_rows)),
                "final_selection": final_selection,
                "holdout_enabled": holdout_window is not None,
                "holdout_metrics": None,
                "diagnostics": diagnostics,
                "data_metadata": panel_data.metadata,
                "warnings": panel_data.warnings,
            }
            write_json(summary, run_dir / "run_summary.json")
            _render_search_report(run_dir / "search_report.md", summary, leaderboard, diagnostics)
            return summary
        except Exception as exc:
            final_selection = fallback_final_selection(
                evidence=final_selection_evidence,
                candidates=center_candidates,
                selector_cfg=final_selector_cfg,
                source="fallback",
                error=f"{type(exc).__name__}: {exc}",
            )
        write_json(final_selection, run_dir / "final_selection.json")
        candidate_by_id = {str(row.get("trial_id")): row for row in center_candidates}
        selected_trial = dict(
            candidate_by_id.get(
                str(final_selection.get("validated_selection", {}).get("selected_trial_id")),
                score_best_trial,
            )
        )
    else:
        pd.DataFrame().to_csv(run_dir / "final_neighbor_trials.csv", index=False)
        pd.DataFrame().to_csv(run_dir / "final_neighbor_window_metrics.csv", index=False)
        write_json(final_selection, run_dir / "final_selection.json")

    validated_final_selection = dict(final_selection.get("validated_selection") or {})
    selected_by = str(validated_final_selection.get("selected_by") or "score_best")
    best_params = {
        "model_type": model_type,
        "selected_by": selected_by,
        "trial_id": selected_trial["trial_id"],
        "score_best_trial_id": score_best_trial["trial_id"],
        "score_best_score": score_best_trial["score"],
        "objective": selected_trial.get("objective"),
        "score": selected_trial["score"],
        "loss": selected_trial["loss"],
        "valid_rmse": selected_trial.get("valid_rmse"),
        "valid_mae": selected_trial.get("valid_mae"),
        "valid_r2": selected_trial.get("valid_r2"),
        "fast_score": selected_trial.get("fast_score"),
        "final_selection": final_selection,
        "params": selected_trial["params"],
    }
    write_json(best_params, run_dir / "best_params.json")

    holdout_summary = None
    if holdout_window is not None:
        holdout_summary, holdout_predictions, holdout_window_metrics = evaluate_holdout(
            panel_data=panel_data,
            holdout_window=holdout_window,
            model_type=model_type,
            params=selected_trial["params"],
            cfg=cfg,
            seed=seed + 9_999_991,
            normalize_method=str(normalize_method) if normalize_method else None,
        )
        write_json(holdout_summary, run_dir / "final_holdout_metrics.json")
        holdout_predictions.to_csv(run_dir / "final_holdout_predictions.csv", index=False)
        holdout_window_metrics.to_csv(run_dir / "final_holdout_window_metrics.csv", index=False)

    summary = {
        "run_id": run_id,
        "task_name": task_name,
        "model_type": model_type,
        "search_method": method,
        "sampler": sampler,
        "probe_fraction": probe_fraction,
        "evaluation_objective": optional_value(cfg, ["evaluation", "objective"], "fast_score"),
        "status": "evaluated",
        "max_trials": max_trials,
        "grid_enabled": use_grid,
        "grid_manifest": grid_manifest,
        "num_grid_trials": len(grid_trials),
        "num_trials_run": num_trials_run,
        "num_successful_trials": num_successful_trials,
        "num_failed_trials": num_failed_trials,
        "num_rounds": int(len(round_history)),
        "llm_enabled": use_llm,
        "decision_provider_type": decision_provider_config.get("type"),
        "decision_provider": dict(decision_provider_config),
        "space_controller_enabled": use_space_controller,
        "space_controller_mode": controller_mode,
        "space_controller_decisions": space_controller_decisions,
        "decision_memory": decision_memory,
        "num_space_versions": int(len(space_versions)),
        "selected_by": best_params["selected_by"],
        "best_trial_id": selected_trial["trial_id"],
        "best_score": selected_trial["score"],
        "best_params": selected_trial["params"],
        "score_best_trial_id": score_best_trial["trial_id"],
        "score_best_score": score_best_trial["score"],
        "score_best_params": score_best_trial["params"],
        "final_selector_enabled": bool(final_selector_cfg.get("enabled")),
        "num_final_neighbor_trials": int(len(final_neighbor_rows)),
        "final_selection": final_selection,
        "holdout_enabled": holdout_window is not None,
        "holdout_metrics": holdout_summary,
        "diagnostics": diagnostics,
        "data_metadata": panel_data.metadata,
        "warnings": panel_data.warnings,
    }
    write_json(summary, run_dir / "run_summary.json")
    _render_search_report(run_dir / "search_report.md", summary, leaderboard, diagnostics)
    return summary
