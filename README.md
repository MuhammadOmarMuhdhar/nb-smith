# nb-smith
CLI-powered agent using nb-cli to manage/audit/clean Jupyter notebooks deterministically. Three sequential phases: inspect structure → quick type fixes → deep diagnostics → full pipeline synthesis. Opinionated, idempotent, "Run-All" safe with asserts.
## Core Idea
CLI-powered agent using nb-cli to manage/audit/clean Jupyter notebooks deterministically. Three sequential phases:
1. inspect structure → 
2. quick type fixes → 
3. deep diagnostics → 
4. full pipeline synthesis
**Opinionated, idempotent, "Run-All" safe with asserts.**
## Four Phases
### Phase 1: Structural Mapping
Scan notebook for DataFrames → profile (head/tail/info/describe/nulls/uniques/shapes) → semantic validation (dates → pd.to_datetime coerce, numbers → pd.to_numeric, sense-checks like negative ages) → insert summary table + suggested fixes cell.
### Phase 2: Quick Type Fixes (Pruning)
Auto-apply safe coercions from Phase 1: fix date strings, numeric strings, strip whitespace, infer categoricals → insert before/after table + dtype asserts → flag new NaNs created.
### Phase 3: Deep Diagnostics (Auditing)
Run on post-fixed data:
- Missingness: %NaNs, MCAR/MAR hints (null corr w/other cols), pattern flags
- Outliers: IQR/Z-score simple rules
- Duplicates, high-cardinality cats, skew, target leakage (corr>0.9)
Output: issue table (type|col|severity|fix_snippet) + asserts
### Phase 4: Pipeline Synthesis (Prep)
Build full cleaning flow from diagnostics:
- Imputation (median/mode/forward-fill per issue type)
- Encoding (one-hot low-card, hash high-card)
- Scaling (RobustScaler post-outliers)
- Feature eng (date→components, bin outliers)
Sequential cells with asserts + final "ready dataset" summary
## Tech Stack
- Typer (Python CLI framework, type-hint simple)
- Pure Python target + nb-cli subprocess calls
- Standalone CLI first (pip install nb-smith)