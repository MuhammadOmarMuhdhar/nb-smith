"""
Shared utilities for smithy modules.

Provides common helpers for notebook cell generation, DataFrame name resolution,
standard output formatting, and severity classification across all phases.
"""

import textwrap
from tools.notebook import insert_cell, read_notebook, insert_cells_batch, find_dataframe_cells

# ── Severity Constants ───────────────────────────────────────────────────────
SEV_LOW = "low"
SEV_MODERATE = "moderate"
SEV_HIGH = "high"

SEVERITY_ORDER = {SEV_HIGH: 0, SEV_MODERATE: 1, SEV_LOW: 2}

# ── Output Formatters ────────────────────────────────────────────────────────
def print_section(title: str) -> str:
    """Return a standard section header string."""
    line = "=" * 70
    return f"\n{line}\n{title}\n{line}"


def tag_info(msg: str) -> str:
    return f"[INFO] {msg}"


def tag_warning(msg: str) -> str:
    return f"[WARNING] {msg}"


def tag_conclusion(msg: str) -> str:
    return msg


def tag_mar(feature: str, target: str, test: str, p: float) -> str:
    return f"MAR: '{target}' ~ '{feature}' ({test}, p_corrected={p:.4f})"


def tag_mnar(col: str) -> str:
    return f"MNAR: '{col}' — no significant associations detected."


# ── DataFrame Name Resolution ────────────────────────────────────────────────
def resolve_df_names(notebook_path: str, df_name: str | None = None) -> list[str]:
    """Resolve which DataFrame(s) to operate on.

    Args:
        notebook_path: Path to the notebook file.
        df_name: Optional explicit DataFrame name. If None, auto-detects all DataFrames.

    Returns:
        List of DataFrame variable names.

    Raises:
        ValueError: If no DataFrames are found in the notebook.
    """
    if df_name:
        return [df_name]

    nb = read_notebook(notebook_path)
    df_cells = find_dataframe_cells(nb)
    df_names = [
        name for _, name in df_cells
        if name and name not in ("null_df", "unique_counts")
    ]

    if not df_names:
        raise ValueError("No DataFrames found in notebook")

    return df_names


# ── Notebook Cell Insertion ──────────────────────────────────────────────────
def insert_analysis_cells(
    notebook_path: str,
    df_names: list[str],
    code_generator: callable,
    label: str,
    call_template: str | None = None,
) -> None:
    """Insert a helper code cell + per-DataFrame analysis cells into a notebook.

    Args:
        notebook_path: Path to the notebook file.
        df_names: List of DataFrame variable names.
        code_generator: Callable that returns the helper code string.
        label: Human-readable label for the diagnostic (e.g. "missingness").
        call_template: Optional custom call template. Use {df} as placeholder.
                       Defaults to "analyze_{label}({df}, name='{df}')".
    """
    helper_code = code_generator()
    insert_cell(notebook_path, helper_code, cell_type="code")

    template = call_template or f"analyze_{label}({{df}}, name='{{df}}')"
    cells_to_insert = [
        {"code": template.format(df=df), "type": "code"}
        for df in df_names
    ]

    insert_cells_batch(notebook_path, cells_to_insert)
    print(f"Added {len(cells_to_insert)} {label} analysis cells")


# ── Code Generation Helpers ──────────────────────────────────────────────────
def dedent_block(code: str) -> str:
    """Dedent a multi-line code block and strip leading/trailing whitespace."""
    return textwrap.dedent(code).strip()


def generate_imports(*names: str) -> str:
    """Generate a block of import statements."""
    lines = []
    for name in names:
        lines.append(f"import {name}")
    return "\n".join(lines)
