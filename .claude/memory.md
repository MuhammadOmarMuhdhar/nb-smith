# nb-smith - Jupyter Notebook Cleaning CLI

## Project Identity
CLI tool for auditing, cleaning, and standardizing Jupyter notebooks deterministically. Philosophy: "Opinionated, idempotent, Run-All safe with asserts." Uses metallurgy metaphor (assay→smelt→cupel→cast→forge).

## Architecture: 4-Phase Pipeline

**Phase 1: ASSAY** (Structural Mapping)
- Scan notebook for DataFrames → profile (head/tail/info/describe/nulls/uniques/shapes)
- Semantic validation: dates→pd.to_datetime, numbers→pd.to_numeric, sense-checks (negative ages)
- Insert summary table + suggested fixes cell

**Phase 2: SMELT** (Quick Type Fixes/Pruning)
- Auto-apply safe coercions: fix date strings, numeric strings, strip whitespace, infer categoricals
- Insert before/after table + dtype asserts
- Flag new NaNs created by coercion

**Phase 3: CUPEL** (Deep Diagnostics/Auditing)
- Missingness: %NaNs, MCAR/MAR hints (null correlation), pattern flags
- Outliers: IQR/Z-score simple rules
- Duplicates, high-cardinality categoricals, skew, target leakage (corr>0.9)
- Output: issue table (type|col|severity|fix_snippet) + asserts

**Phase 4: CAST** (Pipeline Synthesis/Prep)
- Build full cleaning flow: imputation (median/mode/forward-fill), encoding (one-hot/hash), scaling (RobustScaler), feature engineering (date→components, bin outliers)
- Sequential cells with asserts + final "ready dataset" summary

## Module Structure
```
src/
├── cli/
│   └── main.py          # Typer CLI entry (currently stubs)
├── smithy/              # Phase implementations
│   ├── assay.py         # Phase 1 (stubs)
│   ├── smelt.py         # Phase 2 (stubs)
│   ├── cupel.py         # Phase 3 (stubs)
│   └── cast.py          # Phase 4 (stubs)
└── tools/
    ├── notebook.py      # nb-cli wrapper: read/write/insert/execute cells
    └── parsing.py       # (not reviewed)
```

## Tech Stack
- **CLI**: Typer >=0.12.0
- **Data**: pandas >=2.0.0
- **Notebooks**: nbformat (with nb-cli subprocess wrapper)
- **Python**: >=3.10
- **Package**: MIT license, hatchling build

## Current Implementation Status
**Early stage**: Module stubs exist, `notebook.py` utilities implemented. No phase logic yet. CLI commands not wired up.

## Key Design Principles
1. **Deterministic**: Same input → same output, always
2. **Idempotent**: Safe to run multiple times
3. **Assertive**: Insert validation cells, fail fast on violations
4. **Sequential**: Each phase builds on previous (can't skip)
5. **Opinionated**: Auto-fix common issues with sane defaults

## Critical Context from Review (Rating: 7.5/10)

**Strengths:**
- Real problem (notebooks are messy)
- Well-structured phases mirror actual workflows
- Deterministic + assertive approach is production-minded

**Risks:**
- nb-cli dependency (external CLI, fragile) - consider native nbformat instead
- Crowded space (pandas-profiling, great_expectations, cleanlab exist)
- Auto-fixes in Phase 2 could silently corrupt data - needs user review/approval
- Scope unclear: local dev tool vs CI/CD enforcer vs teaching aid?

**Recommended Focus:**
- Primary use case: "Data scientists preparing messy CSVs for modeling"
- Add `--interactive` mode for fix approval
- Build Phase 1-2 excellently before Phase 4 (sklearn pipelines well-trodden)
- Differentiate with semantic type inference (emails, phones, zipcodes)
- Consider pre-commit hook / GitHub Action integration

## Planned CLI Commands
```bash
nb-smith assay <notebook>   # Phase 1
nb-smith smelt <notebook>   # Phase 2
nb-smith cupel <notebook>   # Phase 3
nb-smith cast <notebook>    # Phase 4
nb-smith forge <notebook>   # All phases sequentially
```

## Development Priority
1. Wire up CLI commands in `main.py`
2. Implement Phase 1 (assay) fully
3. Add interactive approval mode
4. Implement Phase 2 (smelt) with safeguards
5. Phase 3-4 later

## Important Notes
- **Don't read .ipynb files** to save tokens (mentioned by user)
- Keep automation balanced with user control (data integrity critical)
- Metallurgy theme is fun but don't let it constrain practical design

## Critical Bug Fixes Applied (2026-04-26)

### Notebook Deletion Bug - FIXED
**Issue**: `execute_notebook()` was sometimes deleting/corrupting notebooks during execution failures.

**Root Cause**: Error handlers in `read_notebook()` and `execute_notebook()` were auto-calling `upgrade_notebook()` on ANY failure, which would overwrite the notebook file even when the error was unrelated to format versions (e.g., kernel errors, execution timeouts).

**Fix**: Removed all automatic upgrade calls from error handlers. Functions now raise clear exceptions without modifying files. `upgrade_notebook()` remains available for explicit user calls only.

### Notebook Version Compatibility - FIXED
**Issues**:
1. Cells had `id` fields (requires nbformat v4.5) but notebooks were v4.2
2. nb-cli adds non-standard `index` property to cells, causing validation failures
3. Missing required `metadata` field in notebook dict

**Fixes Applied** (src/tools/notebook.py:100-144):
- `save_notebook()` now enforces nbformat v4.5 for all notebooks
- Ensures all required fields exist: `metadata`, `cells`, `nbformat`, `nbformat_minor`
- **Cell sanitization**: Removes non-standard properties (like `index`) before validation
- Only allows valid nbformat properties per cell type:
  - Code cells: `cell_type`, `execution_count`, `metadata`, `outputs`, `source`, `id`
  - Markdown/raw: `cell_type`, `metadata`, `source`, `id`

**Result**: Notebooks now save cleanly without validation warnings or fallback to JSON. No more data loss.

### Module Reload Required
After updating notebook.py, active Python sessions need to reload:
```python
import importlib
import tools.notebook
importlib.reload(tools.notebook)
```
