"""Walk-forward model training and prediction."""

from __future__ import annotations

import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config_utils import optional_value
from data_adapter import Window
from model_registry import ModelSpec, create_model
from preprocess import preprocess_panel


def _complexity_penalty(complexity: dict[str, Any]) -> float:
    family = float(complexity.get("model_family_complexity") or 0.0)
    params = float(complexity.get("num_parameters") or 0.0)
    trees = float(complexity.get("num_trees") or 0.0)
    raw = family / 10.0 + min(params / 1_000_000.0, 1.0) + min(trees / 1000.0, 1.0)
    return float(np.clip(raw / 3.0, 0.0, 1.0))


def _uses_lgbm_native_missing(specs: list[ModelSpec]) -> bool:
    return any(spec.type.lower() in {"lgbm", "lightgbm"} for spec in specs)


def _feature_matrix_for_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    spec: ModelSpec,
    cfg: Mapping[str, Any],
) -> pd.DataFrame:
    features = frame[feature_columns]
    if spec.type.lower() in {"lgbm", "lightgbm"}:
        return features
    fill_value = float(optional_value(cfg, ["model_missing", "mlp_fill_value"], 0.0))
    return features.fillna(fill_value)


def _regression_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_arr = np.asarray(y_true, dtype="float64")
    pred_arr = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(y_arr) & np.isfinite(pred_arr)
    if not mask.any():
        return {"rmse": float("nan"), "mae": float("nan"), "r2": float("nan")}
    residuals = y_arr[mask] - pred_arr[mask]
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    ss_res = float(np.dot(residuals, residuals))
    centered = y_arr[mask] - float(np.mean(y_arr[mask]))
    ss_tot = float(np.dot(centered, centered))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {"rmse": rmse, "mae": mae, "r2": float(r2)}


def train_and_predict(
    panel: pd.DataFrame,
    feature_columns: list[str],
    windows: list[Window],
    specs: list[ModelSpec],
    cfg: Mapping[str, Any],
    seed: int,
    normalize_method: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    window_dates = sorted({date for window in windows for date in (*window.train_dates, *window.valid_dates)})
    if not window_dates:
        raise ValueError("No dates are available in training windows")
    panel_for_windows = panel[panel["date"].isin(window_dates)].copy()
    processed = preprocess_panel(
        panel_for_windows,
        feature_columns,
        cfg,
        normalize_method=normalize_method,
        preserve_nan=_uses_lgbm_native_missing(specs),
    )
    rows: list[pd.DataFrame] = []
    window_metric_rows: list[dict[str, Any]] = []
    for window in windows:
        train = processed[processed["date"].isin(window.train_dates)].dropna(subset=["y"]).copy()
        valid = processed[processed["date"].isin(window.valid_dates)].dropna(subset=["y"]).copy()
        if train.empty or valid.empty:
            continue
        y_train = train["y"].astype("float64")
        for spec in specs:
            X_train = _feature_matrix_for_model(train, feature_columns, spec, cfg)
            X_valid = _feature_matrix_for_model(valid, feature_columns, spec, cfg)
            started = time.time()
            model = create_model(spec, seed=seed + window.window_id)
            model.fit(X_train, y_train)
            train_pred = model.predict(X_train)
            valid_pred = model.predict(X_valid)
            elapsed = time.time() - started
            train_corr = pd.Series(train_pred).corr(y_train.reset_index(drop=True), method="spearman")
            valid_corr = pd.Series(valid_pred).corr(valid["y"].reset_index(drop=True), method="spearman")
            train_loss_metrics = _regression_metrics(y_train, train_pred)
            valid_loss_metrics = _regression_metrics(valid["y"], valid_pred)
            overfit_penalty = max(0.0, float(train_corr - valid_corr)) if pd.notna(train_corr) and pd.notna(valid_corr) else 0.0
            complexity = model.complexity()
            c_penalty = _complexity_penalty(complexity)
            pred_frame = valid[["date", "ticker", "y"]].copy()
            pred_frame["prediction"] = valid_pred
            pred_frame["model_name"] = spec.name
            pred_frame["model_type"] = spec.type
            pred_frame["window_id"] = window.window_id
            pred_frame["complexity_penalty"] = c_penalty
            pred_frame["overfit_penalty"] = overfit_penalty
            pred_frame["train_rmse"] = train_loss_metrics["rmse"]
            pred_frame["valid_rmse"] = valid_loss_metrics["rmse"]
            pred_frame["train_mae"] = train_loss_metrics["mae"]
            pred_frame["valid_mae"] = valid_loss_metrics["mae"]
            pred_frame["train_r2"] = train_loss_metrics["r2"]
            pred_frame["valid_r2"] = valid_loss_metrics["r2"]
            rows.append(pred_frame)
            window_metric_rows.append(
                {
                    **window.to_dict(),
                    "model_name": spec.name,
                    "model_type": spec.type,
                    "train_rank_corr": train_corr,
                    "valid_rank_corr": valid_corr,
                    "train_rmse": train_loss_metrics["rmse"],
                    "valid_rmse": valid_loss_metrics["rmse"],
                    "train_mae": train_loss_metrics["mae"],
                    "valid_mae": valid_loss_metrics["mae"],
                    "train_r2": train_loss_metrics["r2"],
                    "valid_r2": valid_loss_metrics["r2"],
                    "overfit_penalty": overfit_penalty,
                    "complexity_penalty": c_penalty,
                    "training_seconds": elapsed,
                    **complexity,
                }
            )
    if not rows:
        raise ValueError("No model predictions were produced")
    predictions = pd.concat(rows, ignore_index=True)
    window_metrics = pd.DataFrame(window_metric_rows)
    return predictions, window_metrics
