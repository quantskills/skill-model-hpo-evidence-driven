---
name: model-hpo-evidence-driven
description: "Run evidence-driven LLM decision or deterministic grid hyperparameter search for quantitative multi-factor models. Use when an agent needs to optimize LGBM or MLP hyperparameters on local factor/label CSV data, let an LLM adapt search spaces from trial evidence through guarded decision files, run budgeted grid search, or export reproducible best-parameter artifacts without factor generation or a full trading backtest."
metadata:
  short-description: Evidence-driven or grid model hyperparameter search
  quantSkills:
    organization: https://github.com/quantskills
    repository: quantskills/skill-model-hpo-evidence-driven
    repository_url: https://github.com/quantskills/skill-model-hpo-evidence-driven
    project_type: skill
    collection: model-hpo
    license: GPL-3.0-only
    category: tooling
    tags: [model-hpo, hyperparameter-search, evidence-driven, grid-search]
    platforms: [codex]
    language: zh-en
    status: active
    validation_level: runnable
    maintainer_type: community
    requires: []
    summary_zh: 使用 trial evidence 驱动的 LLM 搜索空间调整或确定性 grid 搜索优化 LGBM/MLP 多因子模型超参数。
    summary_en: Optimize LGBM/MLP quant factor model hyperparameters with trial-evidence-driven LLM search-space adaptation or deterministic grid search.
---

```json qsh-form
{
  "version": 1,
  "task": {
    "placeholder": "说明本地因子/标签文件、特征列、标签列、切分方式和优化目标，或上传输入配置",
    "required": true
  },
  "fields": [
    {
      "key": "model_type",
      "label": "模型类型",
      "type": "select",
      "default": "lgbm",
      "options": [
        { "value": "lgbm", "label": "LightGBM" },
        { "value": "mlp", "label": "MLP" }
      ]
    },
    {
      "key": "search_method",
      "label": "搜索方法",
      "type": "select",
      "default": "evidence_driven",
      "options": [
        { "value": "evidence_driven", "label": "证据驱动自适应" },
        { "value": "grid", "label": "确定性网格搜索" }
      ]
    },
    {
      "key": "trial_budget",
      "label": "试验预算",
      "type": "number",
      "placeholder": "例如 50"
    }
  ],
  "prompt_template": "{{#task}}任务与材料：\n{{task}}\n\n{{/task}}{{#attachments}}用户上传的材料（已放入工作区）：\n{{attachments}}\n\n{{/attachments}}基于本地因子和标签数据，为 {{model_type}} 执行 {{search_method}} 超参数搜索。{{#trial_budget}}试验预算为 {{trial_budget}}。{{/trial_budget}}先做小规模冒烟测试，再生成可复现的试验清单、排行榜、最佳参数和证据链；不生成新因子，也不把本任务扩展成完整交易回测，输出中文报告。"
}
```

# Model HPO Evidence Driven

Use this skill to run hyperparameter search for LGBM or MLP quantitative multi-factor models on local factor and label files. The skill supports two independent search methods: evidence-driven search-space adaptation and deterministic grid search.

## Core Workflow

1. Read `references/input_schema.md` before preparing the input JSON.
2. Prepare local factor and label CSV/parquet files with one row per `(date, ticker)` observation.
3. Choose `search.model_type`: `lgbm` or `mlp`.
4. Choose `search.method`: `evidence_driven` for adaptive trial-evidence search, or `grid` for deterministic grid search.
5. Run a local smoke test first:

```bash
python scripts/run_hpo_search.py --input examples/hpo_lgbm_smoke.json
```

6. For grid search, run `examples/hpo_lgbm_grid_search.json` or `examples/hpo_mlp_grid_search.json`.
7. For LLM decision search, run `examples/hpo_lgbm_codex_external.json`; the runtime writes evidence and decision templates under the run directory.
8. Read `references/output_contract.md` before consuming generated artifacts.

## Output Contract

Produce a timestamped run directory under `output_root`, including:

- `search_manifest.json`
- `resolved_config.yaml`
- `trials.jsonl`
- `trial_leaderboard.csv`
- `best_params.json`
- `search_report.md`
- `grid_manifest.json` and `grid_trials_resolved.json` when `search.method=grid`
- `codex_decisions/` evidence and decision templates when `decision_provider.type=codex_external`

## References

- Use `references/source_boundary.md` for data, LLM decision, and research boundaries.
- Use `references/input_schema.md` for input fields, search methods, and decision-provider settings.
- Use `references/output_contract.md` for artifact names and result fields.
- Use `references/codex_external_workflow.md` for the LLM decision handoff workflow.
- Use `references/validation_notes.md` for assumptions, checks, limitations, and risk boundaries.
