# Output Contract

The main search writes one timestamped run directory under `output_root`:

```text
outputs/<run_id>/
├── search_manifest.json
├── resolved_config.yaml
├── search_space_versions.json
├── trials.jsonl
├── trial_leaderboard.csv
├── trial_window_metrics.csv
├── round_history.json
├── trial_evidence_round_<round_id>.json
├── trial_evidence_history.json
├── decision_memory.json
├── space_controller_decisions.json
├── failure_modes.json
├── score_best_params.json
├── best_params.json
├── final_selection.json
├── run_summary.json
├── grid_manifest.json                  # when search.method=grid
├── grid_trials_resolved.json           # when search.method=grid
├── final_neighbor_trials.csv           # when the run reaches final selection stage
├── final_neighbor_window_metrics.csv   # when the run reaches final selection stage
├── final_selection_evidence.json       # when final_selector.enabled=true
├── final_holdout_metrics.json          # when a holdout/test window is configured
├── final_holdout_predictions.csv       # when a holdout/test window is configured
├── final_holdout_window_metrics.csv    # when a holdout/test window is configured
├── codex_decisions/                    # when decision_provider.type=codex_external
└── search_report.md
```

Files are created only when the corresponding step is enabled or reached. For example, `space_controller_decisions.json` is empty when the controller does not adapt the space. `final_selection.json` records the final-selector status and decision payload; when the selector is disabled or invalid, `best_params.json` falls back to the validation score-best trial, while `score_best_params.json` records the pure score-best artifact.

## Key Artifacts

`search_manifest.json` contains run metadata:

- `run_id`
- `model_type`
- search method, sampler, trial budget, controller mode
- data metadata and feature column list
- validation/holdout window metadata
- `grid_enabled`, `grid_manifest`, and `num_grid_trials` when `search.method=grid`

`trials.jsonl` contains one JSON row per trial:

- `trial_id`, `round_id`, `model_type`
- `params`
- `status`: `ok` or `failed`
- `score`, `objective`, `loss`
- `valid_rmse`, `valid_mae`, `valid_r2`
- `mean_rankic`, `rankic_ir`
- `top_bottom_spread`, `positive_window_ratio`
- `sample_meta`: sampler/probe details

`trial_leaderboard.csv` is a flattened, sorted view of successful and failed trials.

`trial_window_metrics.csv` contains per-window validation metrics for each trial when window-level metrics are available.

`trial_evidence_round_<round_id>.json` stores the evidence snapshot built after each search round. `trial_evidence_history.json` stores the list of all round evidence snapshots.

`grid_manifest.json` is written when `search.method=grid`. It records:

- grid strategy and source (`search.grid` or `search.grid_trials`)
- full candidate count before the trial budget is applied
- selected trial count, selection policy, seed, and truncation state
- search-space parameter order and generated value counts

`grid_trials_resolved.json` is the exact ordered parameter list evaluated by grid search. Use this file for reproducibility or for replaying the same deterministic grid externally.

`search_space_versions.json` records each accepted search-space version:

- initial space
- later spaces produced by guarded rule decisions or LLM decision files
- changed parameters and reason where available

`space_controller_decisions.json` records the LLM or guarded rule decision after each round:

- raw LLM decision when available
- validated decision after schema and guardrail checks
- accepted/rejected state
- validation errors

`decision_memory.json` links the previous search-space decision to later trial evidence. It is used so the LLM can reason over whether earlier `expand`, `shift`, `narrow`, or `keep` actions helped.

`best_params.json` is the final selected parameter artifact. It includes:

- `model_type`
- selected `trial_id`
- validation `score`
- objective metrics
- selected `params`

When `final_selector.enabled=true`, this file may reflect LLM final selection. Otherwise it is the best validation-score trial.

`score_best_params.json` records the best validation-score trial before any final-selector override.

`final_holdout_metrics.json` reports the selected parameters on the holdout test split when a holdout/test window is configured. Holdout metrics are reporting outputs only; they must not feed back into search-space decisions.

## Offline Final Selection Outputs

`python scripts/run_offline_final_selection.py` writes a separate output directory containing:

- `source_run.json`
- `source_trial_leaderboard.csv`
- `score_best_params.json`
- `final_neighbor_trials.csv`
- `final_neighbor_window_metrics.csv`
- `final_selection_evidence.json`
- `final_selection.json`
- `best_params.json`
- `final_holdout_metrics.json` when a holdout/test window is configured
- `final_holdout_predictions.csv` when a holdout/test window is configured and predictions are requested
- `final_holdout_window_metrics.csv` when a holdout/test window is configured
- `offline_summary.json`

Use offline selection when a completed run should be re-read by an LLM final selector without rerunning the original search rounds.

## LLM Decision Artifacts

When `decision_provider.type=codex_external`, the runtime writes structured handoff files under `codex_decisions/`:

- `round_0000_space_evidence.json`: search-space decision evidence for the external LLM decision step
- `round_0000_space_decision.template.json`: valid response shape
- `round_0000_space_decision.json`: externally written decision file consumed by Python
- `final_selection_evidence.json`: final candidate and neighborhood evidence
- `final_selection_decision.template.json`: final response shape
- `final_selection_decision.json`: externally written final-selection decision, present only after the external LLM/agent writes it

If `on_missing=stop`, the run also writes `external_decision_required.json` and returns `status=external_decision_required`.
