# Model HPO Evidence Driven Skill

**English** | [简体中文](README.md)

> Hyperparameter optimization skill for quant factor models. It supports evidence-driven LLM decision search-space adaptation and deterministic grid search for LGBM and MLP models.

## What It Does

This repository packages a model hyperparameter optimization workflow as a reusable agent skill. It does not compute alpha factors and does not run a production portfolio backtest. It assumes local factor and label files are already prepared.

The runtime records validation metrics, parameters, sampler provenance, and risk signals for each trial. With `search.method=evidence_driven` and `decision_provider.type=codex_external`, the runtime writes structured evidence and decision templates under `codex_decisions/`; an external LLM decision step can then write guarded decision JSON files. With `search.method=grid`, the runtime evaluates explicit `grid_trials` or a budgeted Cartesian grid generated from `search.space`.

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_hpo_search.py --input examples/hpo_lgbm_smoke.json
```

MLP smoke test:

```bash
python scripts/run_hpo_search.py --input examples/hpo_mlp_smoke.json
```

Grid search:

```bash
python scripts/run_hpo_search.py --input examples/hpo_lgbm_grid_search.json
python scripts/run_hpo_search.py --input examples/hpo_mlp_grid_search.json
```

LLM decision mode:

```bash
python scripts/run_hpo_search.py --input examples/hpo_lgbm_codex_external.json
```

## Key Outputs

Each run writes a timestamped output directory containing trial records, leaderboard CSVs, search-space versions, controller decisions, selected parameters, and a markdown search report. Holdout metrics are written only when a holdout/test window is configured. LLM decision runs also write evidence and decision templates under `codex_decisions/`.

See `references/input_schema.md`, `references/codex_external_workflow.md`, and `references/output_contract.md` for the full contract.

## Boundary

This is a research tool for model hyperparameter search. It does not provide investment advice and does not guarantee trading performance.
