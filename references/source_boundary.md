# Source Boundary

This skill operates on user-provided local factor and label files. It does not download market data, compute factor libraries, or call external research APIs.

## Data Boundary

- Inputs are local CSV/parquet files only.
- Factor rows are merged with label rows on `(date, ticker)`.
- The skill assumes labels have already been constructed by the user.
- A missing `available_date` or label-window metadata is reported as a warning, not silently treated as point-in-time safe.

## LLM Boundary

The LLM only receives structured trial evidence, search-space summaries, decision memory, and risk flags. It does not receive raw factor matrices or labels.

The LLM is not allowed to invent arbitrary parameters outside the validated search-space schema. Runtime guardrails validate:

- action names
- changed parameter names
- numeric bounds
- choice values
- maximum expansion ratios
- final selected trial IDs

If LLM output is invalid, the runtime falls back to the guarded rule decision or score-best selection, depending on the stage.

## Research Boundary

This skill is a model hyperparameter optimization harness. It does not provide investment advice, does not claim positive returns, and does not replace a production backtest. Use generated parameters as research candidates that require independent validation.
