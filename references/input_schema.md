# Input Schema

The preferred portable entrypoint is JSON:

```bash
python scripts/run_hpo_search.py --input examples/hpo_lgbm_smoke.json
```

The JSON file may contain wrapper fields plus a native config object:

```json
{
  "output_root": "../outputs/my_run",
  "config": {
    "task": {},
    "input": {},
    "data": {},
    "search": {},
    "validation": {},
    "evaluation": {},
    "space_controller": {},
    "llm": {},
    "final_selector": {}
  }
}
```

Relative paths in `input` and `data` are resolved relative to the JSON file location.

## Required Data Files

`input.feature_path` points to a CSV or parquet file with:

- `date`: date as `YYYYMMDD` integer/string or parseable date string
- ticker column: configured by `data.ticker_col`, for example `symbol`
- numeric factor columns, for example `alpha_001`, `alpha_002`

`input.label_path` points to a CSV or parquet file with:

- same date column
- same ticker column
- label column configured by `data.label_col`, usually `y`

The runtime merges features and labels on `(date, ticker)`. It raises on duplicated keys.

Optional point-in-time metadata:

- `available_date` in factor data: checked to be no later than signal date
- `label_start_date`, `label_end_date` in label data: checked against `time.trade_lag_days`

## Core Config Sections

`data`:

- `date_col`: source date column name
- `ticker_col`: source ticker/asset column name
- `label_col`: label column name in label file
- `start_date`, `end_date`: optional date filter
- `feature_include`: optional explicit factor list
- `feature_exclude`: optional factor exclusions
- `all_null_feature_policy`: `allow`, `drop`, or `raise`
- `compute_hash`: whether to hash input files in metadata

`search`:

- `model_type`: `lgbm` or `mlp`
- `method`: `evidence_driven` or `grid`; legacy aliases `adaptive_tpe` and `fixed_grid` are still accepted
- `sampler`: for `evidence_driven`, use `evidence_probe`, `adaptive`, `structured_probe`, or `local_probe`; for `grid`, this is forced to `grid`
- `max_trials`: total trial budget
- `max_rounds`: maximum search rounds
- `trials_per_round`: trials per round
- `random_start_trials`: random warm-up trials before adaptive sampling; ignored by `grid`
- `top_fraction`: fraction of successful trials used as top evidence; ignored by `grid`
- `space`: search-space mapping used by evidence-driven sampling and generated grid search
- `grid`: generated-grid settings when `method=grid`
- `grid_trials`: optional explicit parameter list when `method=grid`; takes precedence over generated `grid`

Search methods:

- `evidence_driven`: samples parameters from `search.space`, records trial evidence, and optionally lets the decision provider adapt the search space between rounds.
- `grid`: evaluates a deterministic sequence of parameter points. It does not call the decision provider and automatically disables `space_controller`.

Generated grid config:

```json
{
  "method": "grid",
  "max_trials": 12,
  "grid": {
    "strategy": "budgeted_cartesian",
    "numeric_levels": 3,
    "log_levels": 3,
    "choice_policy": "first_middle_last",
    "selection": "evenly_spaced",
    "shuffle": false,
    "seed": 42
  },
  "space": {}
}
```

- `strategy`: currently `budgeted_cartesian`
- `numeric_levels`: number of values generated for `uniform` and `quniform` specs
- `log_levels`: number of values generated for `loguniform` and `qloguniform` specs
- `choice_policy`: `all`, `first_last`, or `first_middle_last`
- `selection`: `evenly_spaced` or `random` when the full Cartesian grid exceeds `max_trials`
- `shuffle`: whether to shuffle selected grid points after deterministic selection
- `seed`: deterministic seed for random selection or shuffle

Search-space spec types:

```json
{"type": "choice", "values": [1, 2, 3]}
{"type": "uniform", "low": 0.1, "high": 1.0}
{"type": "loguniform", "low": 0.0001, "high": 0.01}
{"type": "quniform", "low": 1, "high": 10, "q": 1}
{"type": "qloguniform", "low": 1, "high": 100, "q": 1}
```

`validation` supports:

- `fixed_train_valid_test`: explicit train, validation, and holdout test dates
- `walk_forward`: rolling train/validation windows using trading-day counts

For `fixed_train_valid_test`, define:

- `train_start`, `train_end`
- `valid_start`, `valid_end`
- `test_start`, `test_end`
- `embargo_days`
- `min_assets_per_date`

`evaluation.objective` can be:

- `rankic_ir`: recommended for rank-oriented quant model search
- `rmse`: optimize prediction loss directly
- `fast_score`: weighted composite score


## Decision Provider

`decision_provider` controls who makes search-space and final-selection decisions:

```json
{
  "type": "rule | codex_external",
  "decision_dir": "codex_decisions",
  "wait_for_decision": false,
  "on_missing": "fallback",
  "on_invalid": "fallback"
}
```

- `rule`: deterministic guarded rule decisions.
- `codex_external`: Python writes evidence/template files and reads externally written LLM decision JSON files. This is the file-based LLM decision mode used by the skill.

For `codex_external`, set `on_missing=stop` when the run should pause after evidence generation, or `on_missing=fallback` for non-blocking smoke tests. `decision_provider` is used by evidence-driven search and final selection; `search.method=grid` does not request search-space decisions. This Skill supports `rule` and `codex_external` decision providers only.

`space_controller` controls whether and how the search space is adapted:

- `enabled`: true/false
- `mode`: `rule` or `llm_guarded`
- `exploration_preferred_actions`: usually `expand`, `shift`
- `relocation_preferred_actions`: usually `shift`, `expand`, `narrow`
- `exploitation_preferred_actions`: usually `narrow`, `shift`, `keep`, `stop`

LLM decision settings:

- Set `decision_provider.type` to `codex_external`.
- Keep `llm.enabled=false`; the LLM decision is supplied through the `codex_decisions/` JSON handoff files.
- Use `decision_provider.on_missing=stop` when a run should pause for the external LLM decision step.
- Use `decision_provider.on_missing=fallback` for non-blocking smoke tests.

`final_selector` lets the LLM inspect all completed trials plus small neighborhood probes and select one existing center trial:

- `enabled`: true/false
- `mode`: `llm_guarded`
- `top_k`: number of top candidates shown
- `neighbors_per_candidate`: local probes per candidate
- `max_extra_trials`: cap on neighborhood probes
- `max_score_drop`: guardrail versus score-best candidate
- `fallback`: usually `score_best`
