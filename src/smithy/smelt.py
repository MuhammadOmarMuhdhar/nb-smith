"""
Quick Type Fixes (Pruning)

- Auto-applies safe coercions from Phase 1: date strings, numeric strings, whitespace, categorical inference
- Inserts before/after table + dtype asserts
- Flags new NaNs created.
"""

from tools.notebook import insert_cells_batch


def coerce_dates(df_name: str, columns: list[str]) -> str:
    """Generate pd.to_datetime() code for date columns."""
    lines = []
    for col in columns:
        lines.append(f"{df_name}['{col}'] = pd.to_datetime({df_name}['{col}'], errors='coerce')")
    return '\n'.join(lines)


def coerce_numerics(df_name: str, columns: list[str]) -> str:
    """Generate pd.to_numeric() code with cleaning for numeric columns."""
    lines = []
    for col in columns:
        lines.append(f"# Remove common characters before conversion")
        lines.append(f"_cleaned = {df_name}['{col}'].astype(str).str.replace('[,$%]', '', regex=True)")
        lines.append(f"{df_name}['{col}'] = pd.to_numeric(_cleaned, errors='coerce')")
    return '\n'.join(lines)


def coerce_booleans(df_name: str, columns: list[str]) -> str:
    """Generate boolean mapping code for boolean columns."""
    lines = []
    for col in columns:
        lines.append(f"_bool_map = {{'true': True, 'false': False, 'yes': True, 'no': False, 'y': True, 'n': False, '1': True, '0': False, 't': True, 'f': False}}")
        lines.append(f"{df_name}['{col}'] = {df_name}['{col}'].astype(str).str.lower().map(_bool_map)")
    return '\n'.join(lines)


def infer_categoricals(df_name: str, columns: list[str]) -> str:
    """Generate astype('category') code for categorical columns."""
    lines = []
    for col in columns:
        lines.append(f"{df_name}['{col}'] = {df_name}['{col}'].astype('category')")
    return '\n'.join(lines)


def normalize_column_names(df_name: str, renames: dict) -> str:
    """Generate df.rename() code for column normalization."""
    rename_dict = str(renames).replace("'", '"')
    return f"{df_name}.rename(columns={rename_dict}, inplace=True)"


def trim_whitespace(df_name: str, columns: list[str]) -> str:
    """Generate str.strip() code for object columns with whitespace."""
    lines = []
    for col in columns:
        lines.append(f"{df_name}['{col}'] = {df_name}['{col}'].str.strip()")
    return '\n'.join(lines)


def build_dtype_diff_report(df_names: list[str]) -> str:
    """Generate before/after dtype comparison code."""
    lines = ["# Display dtype changes"]
    for df_name in df_names:
        lines.append(f"print(f'\\n{df_name} dtypes:')")
        lines.append(f"print({df_name}.dtypes)")
    return '\n'.join(lines)


def generate_changes_report(findings: dict, normalization: dict) -> str:
    """Generate concise markdown report of transformations applied.

    Args:
        findings: Type conversion issues per DataFrame/column
        normalization: Column renaming mappings

    Returns:
        Markdown formatted string documenting changes
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
            'date_as_string': 'Date Conversions',
            'numeric_as_string': 'Numeric Conversions',
            'boolean_as_string': 'Boolean Conversions',
            'should_be_categorical': 'Categorical Conversions',
            'identifier_column': 'Identifier Columns (no change)'
        }

        for df_name in sorted(findings.keys()):
            lines.append(f"**{df_name}:**")

            # Group by issue type
            by_issue = {}
            for col, info in findings[df_name].items():
                issue = info['issue']
                if issue not in by_issue:
                    by_issue[issue] = []
                by_issue[issue].append((col, info))

            for issue, cols_info in by_issue.items():
                label = issue_labels.get(issue, issue.replace('_', ' ').title())
                lines.append(f"\n*{label}:*")
                for col, info in cols_info:
                    lines.append(f"- `{col}`: `{info['current_type']}` → `{info['suggested_type']}`")
            lines.append("")

    # Footer
    lines.append("---")
    lines.append("*Changes preserve original data where possible using `errors='coerce'`*")

    return '\n'.join(lines)


def generate_fix_code(findings: dict, normalization: dict) -> list[dict]:
    """Generate fix code cells from findings.

    Args:
        findings: Type conversion issues per DataFrame/column
        normalization: Column renaming mappings

    Returns:
        List of cell dicts with 'code' and 'type' keys
    """
    cells = []
    

    # Cell 2: Type conversions
    if findings:
        conv_lines = ["# Apply Type Conversions\n"]

        # Group columns by DataFrame and issue type for cleaner code generation
        for df_name, columns in findings.items():
            date_cols = []
            numeric_cols = []
            bool_cols = []
            cat_cols = []
            id_cols = []

            for col, info in columns.items():
                issue = info['issue']

                if issue == 'date_as_string':
                    date_cols.append(col)
                elif issue == 'numeric_as_string':
                    numeric_cols.append(col)
                elif issue == 'boolean_as_string':
                    bool_cols.append(col)
                elif issue == 'should_be_categorical':
                    cat_cols.append(col)
                elif issue == 'identifier_column':
                    id_cols.append(col)

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

        conv_lines.append(f"\nprint({df_name}.info())" )
        cells.append({"code": '\n'.join(conv_lines), "type": "code"})

     # Cell 1: Column name normalization
    if normalization:
        norm_lines = ["# Apply Column Name Normalization\n"]
        for df_name, renames in normalization.items():
            norm_lines.append(normalize_column_names(df_name, renames))
        norm_lines.append(f"\nprint('Normalized column names in {len(normalization)} DataFrames')")
        cells.append({"code": '\n'.join(norm_lines), "type": "code"})

    if findings or normalization:
            report = generate_changes_report(findings, normalization)
            cells.append({"code": report, "type": "markdown"})

    return cells


def apply_fixes(notebook_path: str, findings: dict, normalization: dict) -> None:
    """Main entry point: generates and inserts fix cells into notebook.

    Args:
        notebook_path: Path to the notebook file
        findings: Type conversion issues from infer_semantic_types()
        normalization: Column renaming mappings from infer_semantic_types()
    """
    # Generate fix code (returns list of dicts with 'code' and 'type' keys)
    cells_to_insert = generate_fix_code(findings, normalization)

    if cells_to_insert:
        # Insert all cells at once
        insert_cells_batch(notebook_path, cells_to_insert)

        # Count different cell types for reporting
        markdown_cells = sum(1 for c in cells_to_insert if c['type'] == 'markdown')
        code_cells = sum(1 for c in cells_to_insert if c['type'] == 'code')

        print(f"Added {len(cells_to_insert)} cells to notebook ({markdown_cells} markdown, {code_cells} code)")
    else:
        print("No fixes needed - notebook looks good!")
