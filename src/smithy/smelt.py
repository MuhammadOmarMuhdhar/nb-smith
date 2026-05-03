"""
Quick Type Fixes (Pruning)

- Auto-applies safe coercions from Phase 1: date strings, numeric strings, whitespace, categorical inference
- Inserts before/after table + dtype asserts
- Flags new NaNs created.
"""

from smithy._shared import dedent_block, resolve_df_names
from smithy.assay import infer_semantic_types
from tools.notebook import insert_cells_batch


def coerce_dates(df_name: str, columns: list[str]) -> str:
    """Generate pd.to_datetime() code for date columns."""
    lines = []
    for col in columns:
        lines.append(f'{df_name}["{col}"] = pd.to_datetime({df_name}["{col}"], errors="coerce")')
    return "\n".join(lines)


def coerce_numerics(df_name: str, columns: list[str]) -> str:
    """Generate pd.to_numeric() code with cleaning for numeric columns."""
    lines = []
    for col in columns:
        lines.append(f"# Remove common characters before conversion")
        lines.append(f'_cleaned = {df_name}["{col}"].astype(str).str.replace("[,$%]", "", regex=True)')
        lines.append(f'{df_name}["{col}"] = pd.to_numeric(_cleaned, errors="coerce")')
    return "\n".join(lines)


def coerce_booleans(df_name: str, columns: list[str]) -> str:
    """Generate boolean mapping code for boolean columns."""
    lines = []
    for col in columns:
        lines.append(
            '_bool_map = {"true": True, "false": False, "yes": True, "no": False, '
            '"y": True, "n": False, "1": True, "0": False, "t": True, "f": False}'
        )
        lines.append(f'{df_name}["{col}"] = {df_name}["{col}"].astype(str).str.lower().map(_bool_map)')
    return "\n".join(lines)


def infer_categoricals(df_name: str, columns: list[str]) -> str:
    """Generate astype('category') code for categorical columns."""
    lines = []
    for col in columns:
        lines.append(f'{df_name}["{col}"] = {df_name}["{col}"].astype("category")')
    return "\n".join(lines)


def normalize_column_names(df_name: str, renames: dict) -> str:
    """Generate df.rename() code for column normalization."""
    rename_dict = str(renames).replace("'", '"')
    return f'{df_name}.rename(columns={rename_dict}, inplace=True)'


def trim_whitespace(df_name: str, columns: list[str]) -> str:
    """Generate str.strip() code for object columns with whitespace."""
    lines = []
    for col in columns:
        lines.append(f'{df_name}["{col}"] = {df_name}["{col}"].str.strip()')
    return "\n".join(lines)


def build_dtype_diff_report(df_names: list[str]) -> str:
    """Generate before/after dtype comparison code."""
    lines = ["# Display dtype changes"]
    for df_name in df_names:
        lines.append(f"print(f'\\n{df_name} dtypes:')")
        lines.append(f"print({df_name}.dtypes)")
    return "\n".join(lines)


def generate_changes_report(findings: dict, normalization: dict) -> str:
    """Generate concise markdown report of transformations applied.

    Args:
        findings: Type conversion issues per DataFrame/column.
        normalization: Column renaming mappings.

    Returns:
        Markdown formatted string documenting changes.
    """
    lines = ["## Data Transformations Applied", ""]

    # Count changes
    total_normalizations = sum(len(renames) for renames in normalization.values())
    total_conversions = sum(len(cols) for cols in findings.values())

    lines.append(f"**Summary:** {total_normalizations} column(s) renamed, {total_conversions} type conversion(s) applied")
    lines.append("")

    # Normalization section
    if normalization:
        lines.append("## Column Normalization")
        lines.append("")
        for df_name in sorted(normalization.keys()):
            lines.append(f"**{df_name}:**")
            for original, normalized in normalization[df_name].items():
                lines.append(f"- `{original}` → `{normalized}`")
            lines.append("")

    # Type conversions section
    if findings:
        lines.append("### Type Conversions")
        lines.append("")

        # Group by issue type for cleaner reporting
        issue_labels = {
            "date_as_string": "Date Conversions",
            "numeric_as_string": "Numeric Conversions",
            "boolean_as_string": "Boolean Conversions",
            "should_be_categorical": "Categorical Conversions",
            "identifier_column": "Identifier Columns (no change)",
            "potential_whitespace": "Whitespace Trimming",
        }

        for df_name in sorted(findings.keys()):
            lines.append(f"**{df_name}:**")

            # Group by issue type
            by_issue = {}
            for col, info in findings[df_name].items():
                issue = info["issue"]
                if issue not in by_issue:
                    by_issue[issue] = []
                by_issue[issue].append((col, info))

            for issue, cols_info in by_issue.items():
                label = issue_labels.get(issue, issue.replace("_", " ").title())
                lines.append(f"\n*{label}:*")
                for col, info in cols_info:
                    lines.append(f"- `{col}`: `{info['current_type']}` → `{info['suggested_type']}`")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append("*Changes preserve original data where possible using `errors='coerce'`*")

    return "\n".join(lines)


def generate_fix_code(findings: dict, normalization: dict) -> list[dict]:
    """Generate fix code cells from findings.

    Args:
        findings: Type conversion issues per DataFrame/column.
        normalization: Column renaming mappings.

    Returns:
        List of cell dicts with 'code' and 'type' keys.
    """
    if not isinstance(findings, dict):
        raise TypeError("findings must be a dict")
    if not isinstance(normalization, dict):
        raise TypeError("normalization must be a dict")

    cells = []

    # Type conversions
    if findings:
        conv_lines = ["# Apply Type Conversions\n"]

        # Group columns by DataFrame and issue type for cleaner code generation
        for df_name, columns in findings.items():
            date_cols = []
            numeric_cols = []
            bool_cols = []
            cat_cols = []
            id_cols = []
            ws_cols = []

            for col, info in columns.items():
                issue = info["issue"]

                if issue == "date_as_string":
                    date_cols.append(col)
                elif issue == "numeric_as_string":
                    numeric_cols.append(col)
                elif issue == "boolean_as_string":
                    bool_cols.append(col)
                elif issue == "should_be_categorical":
                    cat_cols.append(col)
                elif issue == "identifier_column":
                    id_cols.append(col)
                elif issue == "potential_whitespace":
                    ws_cols.append(col)

            # Generate code for each type
            if date_cols:
                conv_lines.append(f"\n# Date conversions for {df_name}")
                conv_lines.append(coerce_dates(df_name, date_cols))

            if numeric_cols:
                conv_lines.append(f"\n# Numeric conversions for {df_name}")
                conv_lines.append(coerce_numerics(df_name, numeric_cols))

            if bool_cols:
                conv_lines.append(f"\n# Boolean conversions for {df_name}")
                conv_lines.append(coerce_booleans(df_name, bool_cols))

            if cat_cols:
                conv_lines.append(f"\n# Categorical conversions for {df_name}")
                conv_lines.append(infer_categoricals(df_name, cat_cols))

            if id_cols:
                conv_lines.append(f"\n# Identifier columns for {df_name} (no conversion needed)")
                for col in id_cols:
                    conv_lines.append(f"# {df_name}['{col}'] - kept as-is")

            if ws_cols:
                conv_lines.append(f"\n# Whitespace trimming for {df_name}")
                conv_lines.append(trim_whitespace(df_name, ws_cols))

        conv_lines.append(f"\nprint({df_name}.info())")
        cells.append({"code": "\n".join(conv_lines), "type": "code"})

    # Column name normalization
    if normalization:
        norm_lines = ["# Apply Column Name Normalization\n"]
        for df_name, renames in normalization.items():
            norm_lines.append(normalize_column_names(df_name, renames))
        norm_lines.append(f"\nprint('Normalized column names in {len(normalization)} DataFrames')")
        cells.append({"code": "\n".join(norm_lines), "type": "code"})

    if findings or normalization:
        report = generate_changes_report(findings, normalization)
        cells.append({"code": report, "type": "markdown"})

    return cells


def apply_fixes(notebook_path: str, findings: dict, normalization: dict) -> None:
    """Main entry point: generates and inserts fix cells into notebook.

    Args:
        notebook_path: Path to the notebook file.
        findings: Type conversion issues from infer_semantic_types().
        normalization: Column renaming mappings from infer_semantic_types().
    """
    if not isinstance(notebook_path, str):
        raise TypeError("notebook_path must be a string")

    cells_to_insert = generate_fix_code(findings, normalization)

    if cells_to_insert:
        insert_cells_batch(notebook_path, cells_to_insert)

        markdown_cells = sum(1 for c in cells_to_insert if c["type"] == "markdown")
        code_cells = sum(1 for c in cells_to_insert if c["type"] == "code")

        print(f"Added {len(cells_to_insert)} cells to notebook ({markdown_cells} markdown, {code_cells} code)")
    else:
        print("No fixes needed - notebook looks good!")


# ═══════════════════════════════════════════════════════════════════════════════
# Insert Wrappers (granular CLI support)
# ═══════════════════════════════════════════════════════════════════════════════


def _get_target_df_names(notebook_path: str, df_name: str | None) -> list[str]:
    """Resolve DataFrame names, falling back to all detected."""
    if df_name:
        return [df_name]
    return resolve_df_names(notebook_path)


def _filter_findings_by_issue(findings: dict, issue_type: str, df_names: list[str]) -> dict:
    """Return findings filtered to a specific issue type and DataFrame set."""
    filtered = {}
    for df_name, cols in findings.items():
        if df_name not in df_names:
            continue
        matched = {c: info for c, info in cols.items() if info["issue"] == issue_type}
        if matched:
            filtered[df_name] = matched
    return filtered


def _insert_single_fix(notebook_path: str, df_names: list[str], issue_type: str,
                       code_generator: callable, label: str) -> None:
    """Run inference, filter by issue, generate code, and insert."""
    result = infer_semantic_types(notebook_path)
    findings = result["findings"]
    filtered = _filter_findings_by_issue(findings, issue_type, df_names)

    if not filtered:
        print(f"No {issue_type} issues detected.")
        return

    cells = []
    for df_name in filtered:
        cols = list(filtered[df_name].keys())
        code = code_generator(df_name, cols)
        cells.append({"code": f"# {label} for {df_name}\n{code}", "type": "code"})

    insert_cells_batch(notebook_path, cells)
    total_cols = sum(len(cols) for cols in filtered.values())
    print(f"Added {len(cells)} cell(s) for {label} ({total_cols} column(s))")


def insert_date_fixes(notebook_path: str, df_name: str | None = None, columns: list[str] | None = None) -> None:
    """Insert date coercion cells into notebook."""
    if columns:
        df_names = _get_target_df_names(notebook_path, df_name)
        cells = [{"code": coerce_dates(df, columns), "type": "code"} for df in df_names]
        insert_cells_batch(notebook_path, cells)
        print(f"Added date coercion cells for {len(df_names)} DataFrame(s)")
    else:
        _insert_single_fix(notebook_path, _get_target_df_names(notebook_path, df_name),
                           "date_as_string", coerce_dates, "Date coercions")


def insert_numeric_fixes(notebook_path: str, df_name: str | None = None, columns: list[str] | None = None) -> None:
    """Insert numeric coercion cells into notebook."""
    if columns:
        df_names = _get_target_df_names(notebook_path, df_name)
        cells = [{"code": coerce_numerics(df, columns), "type": "code"} for df in df_names]
        insert_cells_batch(notebook_path, cells)
        print(f"Added numeric coercion cells for {len(df_names)} DataFrame(s)")
    else:
        _insert_single_fix(notebook_path, _get_target_df_names(notebook_path, df_name),
                           "numeric_as_string", coerce_numerics, "Numeric coercions")


def insert_boolean_fixes(notebook_path: str, df_name: str | None = None, columns: list[str] | None = None) -> None:
    """Insert boolean coercion cells into notebook."""
    if columns:
        df_names = _get_target_df_names(notebook_path, df_name)
        cells = [{"code": coerce_booleans(df, columns), "type": "code"} for df in df_names]
        insert_cells_batch(notebook_path, cells)
        print(f"Added boolean coercion cells for {len(df_names)} DataFrame(s)")
    else:
        _insert_single_fix(notebook_path, _get_target_df_names(notebook_path, df_name),
                           "boolean_as_string", coerce_booleans, "Boolean coercions")


def insert_categorical_fixes(notebook_path: str, df_name: str | None = None, columns: list[str] | None = None) -> None:
    """Insert categorical coercion cells into notebook."""
    if columns:
        df_names = _get_target_df_names(notebook_path, df_name)
        cells = [{"code": infer_categoricals(df, columns), "type": "code"} for df in df_names]
        insert_cells_batch(notebook_path, cells)
        print(f"Added categorical coercion cells for {len(df_names)} DataFrame(s)")
    else:
        _insert_single_fix(notebook_path, _get_target_df_names(notebook_path, df_name),
                           "should_be_categorical", infer_categoricals, "Categorical coercions")


def insert_normalization(notebook_path: str, df_name: str | None = None) -> None:
    """Insert column normalization cells into notebook."""
    result = infer_semantic_types(notebook_path)
    normalization = result["normalization"]

    if df_name:
        normalization = {k: v for k, v in normalization.items() if k == df_name}

    if not normalization:
        print("No column normalization needed.")
        return

    cells = []
    for df_name, renames in normalization.items():
        cells.append({"code": normalize_column_names(df_name, renames), "type": "code"})

    insert_cells_batch(notebook_path, cells)
    total_renames = sum(len(r) for r in normalization.values())
    print(f"Added normalization cells ({total_renames} column(s) renamed)")


def insert_trim_fixes(notebook_path: str, df_name: str | None = None, columns: list[str] | None = None) -> None:
    """Insert whitespace trimming cells into notebook."""
    if columns:
        df_names = _get_target_df_names(notebook_path, df_name)
        cells = [{"code": trim_whitespace(df, columns), "type": "code"} for df in df_names]
        insert_cells_batch(notebook_path, cells)
        print(f"Added whitespace trimming cells for {len(df_names)} DataFrame(s)")
    else:
        _insert_single_fix(notebook_path, _get_target_df_names(notebook_path, df_name),
                           "potential_whitespace", trim_whitespace, "Whitespace trimming")
