"""Data loading, schema validation, and walk-forward windows for model research."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from config_utils import ConfigError, file_sha256, optional_value, require_mapping, require_value, resolve_path


@dataclass(frozen=True)
class Window:
    window_id: int
    train_start: int
    train_end: int
    valid_start: int
    valid_end: int
    train_dates: tuple[int, ...]
    valid_dates: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["train_dates"] = list(self.train_dates)
        out["valid_dates"] = list(self.valid_dates)
        return out


@dataclass
class PanelData:
    panel: pd.DataFrame
    feature_columns: list[str]
    warnings: list[str]
    metadata: dict[str, Any]


def read_table(
    path: Path,
    *,
    date_col: str | None = None,
    start_date: int | str | None = None,
    end_date: int | str | None = None,
    max_rows: int | None = None,
    chunksize: int = 500_000,
    usecols: list[str] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        read_kwargs: dict[str, Any] = {}
        if usecols:
            header = pd.read_csv(path, nrows=0).columns.astype(str).tolist()
            selected = [col for col in dict.fromkeys(usecols) if col in header]
            if selected:
                read_kwargs["usecols"] = selected
        if not date_col or (start_date in (None, "") and end_date in (None, "") and not max_rows):
            return pd.read_csv(path, **read_kwargs)
        frames: list[pd.DataFrame] = []
        kept = 0
        start_value = int(start_date) if start_date not in (None, "") else None
        end_value = int(end_date) if end_date not in (None, "") else None
        for chunk in pd.read_csv(path, chunksize=chunksize, **read_kwargs):
            if date_col not in chunk.columns:
                raise ConfigError(f"Input file is missing configured date column {date_col!r}: {path}")
            dates = normalize_date_series(chunk[date_col])
            mask = pd.Series(True, index=chunk.index)
            if start_value is not None:
                mask &= dates >= start_value
            if end_value is not None:
                mask &= dates <= end_value
            out = chunk.loc[mask].copy()
            if out.empty:
                continue
            if max_rows is not None:
                remaining = int(max_rows) - kept
                if remaining <= 0:
                    break
                out = out.head(remaining)
            frames.append(out)
            kept += len(out)
            if max_rows is not None and kept >= int(max_rows):
                break
        if not frames:
            return pd.read_csv(path, nrows=0)
        return pd.concat(frames, ignore_index=True)
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path, columns=list(dict.fromkeys(usecols)) if usecols else None)
        if date_col and (start_date not in (None, "") or end_date not in (None, "")):
            dates = normalize_date_series(frame[date_col])
            mask = pd.Series(True, index=frame.index)
            if start_date not in (None, ""):
                mask &= dates >= int(start_date)
            if end_date not in (None, ""):
                mask &= dates <= int(end_date)
            frame = frame.loc[mask].copy()
        if max_rows is not None:
            frame = frame.head(int(max_rows)).copy()
        return frame
    raise ConfigError(f"Unsupported table format {suffix!r}; expected .csv or .parquet")


def normalize_date_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_integer_dtype(series):
        return series.astype("int64")
    if pd.api.types.is_float_dtype(series):
        if not np.all(np.isfinite(series.dropna())):
            raise ValueError("Date column contains non-finite values")
        return series.astype("int64")
    text = series.astype(str).str.strip()
    numeric_mask = text.str.fullmatch(r"\d{8}")
    result = pd.Series(index=series.index, dtype="int64")
    if numeric_mask.any():
        result.loc[numeric_mask] = text.loc[numeric_mask].astype("int64")
    if (~numeric_mask).any():
        parsed = pd.to_datetime(text.loc[~numeric_mask], errors="raise")
        result.loc[~numeric_mask] = parsed.dt.strftime("%Y%m%d").astype("int64")
    return result.astype("int64")


def normalize_ticker_series(series: pd.Series, ticker_dtype: str) -> pd.Series:
    if ticker_dtype == "preserve":
        return series
    if ticker_dtype == "str":
        return series.astype(str).str.strip()
    if ticker_dtype == "int":
        numeric = pd.to_numeric(series, errors="raise")
        if numeric.isna().any():
            raise ValueError("Ticker column contains missing values and cannot be converted to int")
        return numeric.astype("int64")
    raise ConfigError("data.ticker_dtype must be one of: preserve, str, int")


def _standardize_key_columns(df: pd.DataFrame, cfg: Mapping[str, Any], table_name: str) -> pd.DataFrame:
    data_cfg = require_mapping(cfg, ["data"])
    date_col = require_value(cfg, ["data", "date_col"])
    ticker_col = data_cfg.get("ticker_col") or data_cfg.get("asset_col")
    if not ticker_col:
        raise ConfigError("Config must define data.ticker_col or data.asset_col")
    if date_col not in df.columns:
        raise ConfigError(f"{table_name} is missing date column: {date_col}")
    if ticker_col not in df.columns:
        raise ConfigError(f"{table_name} is missing ticker column: {ticker_col}")
    out = df.copy()
    out["date"] = normalize_date_series(out[date_col])
    out["ticker"] = normalize_ticker_series(out[ticker_col], str(optional_value(cfg, ["data", "ticker_dtype"], "preserve")))
    if date_col != "date":
        out = out.drop(columns=[date_col])
    if ticker_col != "ticker":
        out = out.drop(columns=[ticker_col])
    return out


def _validate_unique_key(df: pd.DataFrame, table_name: str) -> None:
    duplicates = df.duplicated(["date", "ticker"]).sum()
    if duplicates:
        raise ValueError(f"{table_name} contains duplicated (date, ticker) rows: {duplicates}")


def _select_feature_columns(features: pd.DataFrame, cfg: Mapping[str, Any]) -> list[str]:
    data_cfg = require_mapping(cfg, ["data"])
    include = list(data_cfg.get("feature_include") or [])
    exclude = set(data_cfg.get("feature_exclude") or [])
    reserved = {
        "date",
        "ticker",
        "available_date",
        "industry",
        "market_cap",
        "in_universe",
        str(data_cfg.get("label_col", "y")),
    }
    if include:
        missing = [col for col in include if col not in features.columns]
        if missing:
            raise ConfigError(f"Configured feature_include columns are missing: {missing}")
        non_numeric = [col for col in include if not pd.api.types.is_numeric_dtype(features[col])]
        if non_numeric:
            raise ConfigError(f"Configured feature_include columns are not numeric: {non_numeric}")
        feature_cols = include
    else:
        feature_cols = [
            col for col in features.columns
            if col not in reserved and pd.api.types.is_numeric_dtype(features[col])
        ]
    feature_cols = [col for col in feature_cols if col not in exclude]
    if not feature_cols:
        raise ConfigError("No numeric feature columns are available after include/exclude filtering")
    return feature_cols


def _check_available_date(features: pd.DataFrame, warnings: list[str]) -> None:
    if "available_date" not in features.columns:
        warnings.append("point_in_time_risk: available_date column is missing")
        return
    available = normalize_date_series(features["available_date"])
    bad = (available > features["date"]).sum()
    if bad:
        raise ValueError(f"available_date is after signal date for {bad} rows")


def _check_label_window(labels: pd.DataFrame, warnings: list[str], trade_lag_days: int) -> None:
    if trade_lag_days < 0:
        raise ConfigError("time.trade_lag_days must be non-negative")
    if "label_start_date" not in labels.columns or "label_end_date" not in labels.columns:
        warnings.append("label_window_risk: label_start_date/label_end_date columns are missing")
        return
    feature_date = normalize_date_series(labels["date"])
    start = normalize_date_series(labels["label_start_date"])
    end = normalize_date_series(labels["label_end_date"])
    if (start > end).any():
        raise ValueError("label_start_date is after label_end_date")
    if (start < feature_date).any():
        raise ValueError("label_start_date is before feature date")
    if trade_lag_days:
        calendar = sorted(set(feature_date.tolist()) | set(start.tolist()) | set(end.tolist()))
        rank = {value: idx for idx, value in enumerate(calendar)}
        feature_rank = feature_date.map(rank)
        start_rank = start.map(rank)
        bad = (start_rank < feature_rank + trade_lag_days).sum()
        if bad:
            raise ValueError(
                "label_start_date violates time.trade_lag_days "
                f"for {int(bad)} rows"
            )


def build_panel(cfg: Mapping[str, Any]) -> PanelData:
    base_dir = cfg.get("_config_dir", ".")
    feature_path = resolve_path(require_value(cfg, ["data", "feature_path"]), base_dir)
    label_path = resolve_path(require_value(cfg, ["data", "label_path"]), base_dir)
    universe_path = resolve_path(optional_value(cfg, ["data", "universe_path"]), base_dir)
    label_col = str(require_value(cfg, ["data", "label_col"]))

    warnings: list[str] = []
    data_cfg = require_mapping(cfg, ["data"])
    date_col = str(require_value(cfg, ["data", "date_col"]))
    ticker_col = str(data_cfg.get("ticker_col") or data_cfg.get("asset_col") or "ticker")
    feature_include = list(data_cfg.get("feature_include") or [])
    feature_usecols = [date_col, ticker_col, "available_date", *feature_include] if feature_include else None
    label_usecols = [date_col, ticker_col, label_col, "label_start_date", "label_end_date"]
    start_date = optional_value(cfg, ["data", "start_date"])
    end_date = optional_value(cfg, ["data", "end_date"])
    max_rows = optional_value(cfg, ["data", "max_rows"])
    chunksize = int(optional_value(cfg, ["data", "read_chunksize"], 500_000))
    features = _standardize_key_columns(
        read_table(
            feature_path,
            date_col=date_col,
            start_date=start_date,
            end_date=end_date,
            max_rows=int(max_rows) if max_rows not in (None, "") else None,
            chunksize=chunksize,
            usecols=feature_usecols,
        ),
        cfg,
        "feature_df",
    )
    labels = _standardize_key_columns(
        read_table(
            label_path,
            date_col=date_col,
            start_date=start_date,
            end_date=end_date,
            max_rows=int(max_rows) if max_rows not in (None, "") else None,
            chunksize=chunksize,
            usecols=label_usecols,
        ),
        cfg,
        "label_df",
    )
    _validate_unique_key(features, "feature_df")
    _validate_unique_key(labels, "label_df")
    if label_col in features.columns:
        raise ConfigError("feature_df must not contain the configured label_col")
    if label_col not in labels.columns:
        raise ConfigError(f"label_df is missing configured label_col: {label_col}")
    labels = labels.rename(columns={label_col: "y"})

    _check_available_date(features, warnings)
    trade_lag_days = int(optional_value(cfg, ["time", "trade_lag_days"], 0))
    _check_label_window(labels, warnings, trade_lag_days)
    feature_cols = _select_feature_columns(features, cfg)

    keep_label_cols = ["date", "ticker", "y"]
    for optional_col in ("label_start_date", "label_end_date"):
        if optional_col in labels.columns:
            keep_label_cols.append(optional_col)
    panel = features.merge(labels[keep_label_cols], on=["date", "ticker"], how="inner")
    if panel.empty:
        raise ValueError("Merged panel is empty; feature_df and label_df have no shared (date, ticker) rows")

    universe_policy = "inferred_intersection"
    if universe_path:
        universe = _standardize_key_columns(read_table(universe_path), cfg, "universe_df")
        _validate_unique_key(universe, "universe_df")
        keep_cols = [col for col in universe.columns if col not in panel.columns or col in {"date", "ticker"}]
        panel = panel.merge(universe[keep_cols], on=["date", "ticker"], how="left")
        if "in_universe" in panel.columns:
            panel = panel[panel["in_universe"].fillna(False).astype(bool)].copy()
            universe_policy = "filter_in_universe_true"
        else:
            universe_policy = "metadata_joined"
    else:
        warnings.append("universe_risk: universe_df is missing; using feature/label intersection")
    if panel.empty:
        raise ValueError("Panel is empty after universe filtering")

    all_null_feature_cols = [col for col in feature_cols if panel[col].isna().all()]
    all_null_policy = str(data_cfg.get("all_null_feature_policy", "allow")).lower()
    if all_null_policy not in {"allow", "drop", "raise"}:
        raise ConfigError("data.all_null_feature_policy must be one of: allow, drop, raise")
    if all_null_feature_cols:
        preview = all_null_feature_cols[:20]
        suffix = "" if len(all_null_feature_cols) <= 20 else f", ... (+{len(all_null_feature_cols) - 20} more)"
        message = (
            f"all_null_feature_policy={all_null_policy}: "
            f"{len(all_null_feature_cols)} feature columns are all-null after merge: {preview}{suffix}"
        )
        if all_null_policy == "raise":
            raise ValueError(message)
        if all_null_policy == "drop":
            drop_set = set(all_null_feature_cols)
            feature_cols = [col for col in feature_cols if col not in drop_set]
            if not feature_cols:
                raise ValueError("No feature columns remain after dropping all-null features")
        warnings.append(message)

    metadata = {
        "feature_path": str(feature_path),
        "feature_sha256": file_sha256(feature_path) if bool(optional_value(cfg, ["data", "compute_hash"], True)) else None,
        "label_path": str(label_path),
        "label_sha256": file_sha256(label_path) if bool(optional_value(cfg, ["data", "compute_hash"], True)) else None,
        "universe_path": str(universe_path) if universe_path else None,
        "universe_policy": universe_policy,
        "num_rows": int(len(panel)),
        "num_dates": int(panel["date"].nunique()),
        "num_tickers": int(panel["ticker"].nunique()),
        "num_all_null_features": int(len(all_null_feature_cols)),
        "all_null_feature_columns": all_null_feature_cols,
        "all_null_feature_policy": all_null_policy,
    }
    return PanelData(panel=panel, feature_columns=feature_cols, warnings=warnings, metadata=metadata)


def _normalize_config_date(value: Any, name: str) -> int:
    if value in (None, ""):
        raise ConfigError(f"validation.{name} is required")
    return int(normalize_date_series(pd.Series([value])).iloc[0])


def _dates_between(dates: list[int], start: int, end: int, name: str) -> tuple[int, ...]:
    selected = tuple(d for d in dates if start <= d <= end)
    if not selected:
        raise ValueError(f"No panel dates found for {name}: {start}-{end}")
    return selected


def _validate_embargo(cfg: Mapping[str, Any], embargo_n: int) -> None:
    label_window = int(optional_value(cfg, ["training", "label_window"], 0))
    trade_lag = int(optional_value(cfg, ["time", "trade_lag_days"], 0))
    if embargo_n < label_window + trade_lag:
        raise ConfigError("validation.embargo_days must be >= training.label_window + time.trade_lag_days")


def _build_fixed_train_valid_window(panel: pd.DataFrame, cfg: Mapping[str, Any], validation_cfg: Mapping[str, Any]) -> list[Window]:
    dates = sorted(int(x) for x in panel["date"].dropna().unique())
    train_start = _normalize_config_date(validation_cfg.get("train_start"), "train_start")
    train_end = _normalize_config_date(validation_cfg.get("train_end"), "train_end")
    valid_start = _normalize_config_date(validation_cfg.get("valid_start"), "valid_start")
    valid_end = _normalize_config_date(validation_cfg.get("valid_end"), "valid_end")
    embargo_n = int(validation_cfg.get("embargo_days", 0))
    _validate_embargo(cfg, embargo_n)
    if not train_start <= train_end < valid_start <= valid_end:
        raise ConfigError(
            "fixed_train_valid_test requires train_start <= train_end < valid_start <= valid_end"
        )
    train_dates = _dates_between(dates, train_start, train_end, "train split")
    valid_dates = _dates_between(dates, valid_start, valid_end, "validation split")
    gap_dates = [d for d in dates if train_end < d < valid_start]
    if len(gap_dates) < embargo_n:
        raise ConfigError(
            "fixed_train_valid_test split violates validation.embargo_days: "
            f"gap_dates={len(gap_dates)}, required={embargo_n}"
        )
    return [
        Window(
            window_id=0,
            train_start=train_dates[0],
            train_end=train_dates[-1],
            valid_start=valid_dates[0],
            valid_end=valid_dates[-1],
            train_dates=train_dates,
            valid_dates=valid_dates,
        )
    ]


def build_holdout_test_window(panel: pd.DataFrame, cfg: Mapping[str, Any]) -> Window | None:
    validation_cfg = require_mapping(cfg, ["validation"])
    if validation_cfg.get("method") != "fixed_train_valid_test":
        return None
    dates = sorted(int(x) for x in panel["date"].dropna().unique())
    train_start = _normalize_config_date(validation_cfg.get("train_start"), "train_start")
    valid_end = _normalize_config_date(validation_cfg.get("valid_end"), "valid_end")
    test_start = _normalize_config_date(validation_cfg.get("test_start"), "test_start")
    test_end = _normalize_config_date(validation_cfg.get("test_end"), "test_end")
    embargo_n = int(validation_cfg.get("embargo_days", 0))
    if not valid_end < test_start <= test_end:
        raise ConfigError("fixed_train_valid_test requires valid_end < test_start <= test_end")
    gap_dates = [d for d in dates if valid_end < d < test_start]
    if len(gap_dates) < embargo_n:
        raise ConfigError(
            "fixed_train_valid_test holdout split violates validation.embargo_days: "
            f"gap_dates={len(gap_dates)}, required={embargo_n}"
        )
    train_dates = _dates_between(dates, train_start, valid_end, "final train+valid split")
    test_dates = _dates_between(dates, test_start, test_end, "holdout test split")
    return Window(
        window_id=0,
        train_start=train_dates[0],
        train_end=train_dates[-1],
        valid_start=test_dates[0],
        valid_end=test_dates[-1],
        train_dates=train_dates,
        valid_dates=test_dates,
    )


def build_windows(panel: pd.DataFrame, cfg: Mapping[str, Any]) -> list[Window]:
    validation_cfg = require_mapping(cfg, ["validation"])
    method = str(validation_cfg.get("method"))
    if method == "fixed_train_valid_test":
        return _build_fixed_train_valid_window(panel, cfg, validation_cfg)
    if method != "walk_forward":
        raise ConfigError("validation.method must be one of: walk_forward, fixed_train_valid_test")
    if validation_cfg.get("window_unit") != "trading_days":
        raise ConfigError("validation.window_unit must be explicitly set to trading_days")

    train_n = int(require_value(cfg, ["validation", "train_window_days"]))
    valid_n = int(require_value(cfg, ["validation", "valid_window_days"]))
    step_n = int(require_value(cfg, ["validation", "step_days"]))
    embargo_n = int(require_value(cfg, ["validation", "embargo_days"]))
    _validate_embargo(cfg, embargo_n)
    for name, value in {
        "train_window_days": train_n,
        "valid_window_days": valid_n,
        "step_days": step_n,
        "embargo_days": embargo_n,
    }.items():
        if value < 0 or (name != "embargo_days" and value == 0):
            raise ConfigError(f"validation.{name} has invalid value: {value}")

    dates = sorted(int(x) for x in panel["date"].dropna().unique())
    locked_test_start = optional_value(cfg, ["time", "locked_test_start"])
    if locked_test_start not in (None, ""):
        locked_test_start = int(normalize_date_series(pd.Series([locked_test_start])).iloc[0])
        dates = [d for d in dates if d < locked_test_start]
    if len(dates) < train_n + embargo_n + valid_n:
        raise ValueError(
            "Not enough dates to build one walk-forward window: "
            f"num_dates={len(dates)}, required={train_n + embargo_n + valid_n}"
        )
    windows: list[Window] = []
    start = 0
    window_id = 0
    while start + train_n + embargo_n + valid_n <= len(dates):
        train_dates = tuple(dates[start : start + train_n])
        valid_start_idx = start + train_n + embargo_n
        valid_dates = tuple(dates[valid_start_idx : valid_start_idx + valid_n])
        windows.append(
            Window(
                window_id=window_id,
                train_start=train_dates[0],
                train_end=train_dates[-1],
                valid_start=valid_dates[0],
                valid_end=valid_dates[-1],
                train_dates=train_dates,
                valid_dates=valid_dates,
            )
        )
        window_id += 1
        start += step_n
    if not windows:
        raise ValueError("No walk-forward windows were built")
    return windows
