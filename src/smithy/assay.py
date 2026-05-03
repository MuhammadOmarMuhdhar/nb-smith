"""
Structural Mapping

- Scans notebook for DataFrames → profiles (head/tail/info/describe/nulls/uniques/shapes)
- Semantic validation (dates → pd.to_datetime, numbers → pd.to_numeric, sense-checks)
- Inserts summary table of all that was done.
"""

import re
import textwrap
from collections import defaultdict
from pathlib import Path

import openpyxl

from smithy._shared import dedent_block, insert_analysis_cells, resolve_df_names
from tools.notebook import create_notebook, find_dataframe_cells, insert_cell, read_notebook


PANDAS_READERS = {
    ".csv":      "csv",
    ".xlsx":     "excel",
    ".xls":      "excel",
    ".xlsm":     "excel",
    ".json":     "json",
    ".parquet":  "parquet",
    ".html":     "html",
    ".sql":      "sql",
    ".xml":      "xml",
    ".feather":  "feather",
    ".dta":      "stata",
    ".sas7bdat": "sas",
    ".pkl":      "pickle",
}

TYPE_LABELS = {
    "csv":     "CSV files",
    "excel":   "Excel files",
    "json":    "JSON files",
    "parquet": "Parquet files",
    "feather": "Feather files",
    "stata":   "Stata files",
    "sas":     "SAS files",
    "pickle":  "Pickle files",
}


def _sanitize(name: str) -> str:
    """Sanitize a string for use as a Python variable name."""
    return name.replace(" ", "_").replace("-", "_")


def _get_excel_sheets(path: Path) -> list[str]:
    """Return all sheet names in an Excel file."""
    try:
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        sheets = wb.sheetnames
        wb.close()
        return sheets
    except Exception:
        return []


def _collect_data_files(directory: Path) -> dict[str, list[dict]]:
    """Return data files grouped by pandas reader type.

    Excel files with multiple sheets are expanded into one entry per sheet.
    """
    files_by_type = defaultdict(list)

    for item in sorted(directory.iterdir()):
        if not item.is_file():
            continue
        reader = PANDAS_READERS.get(item.suffix.lower())
        if not reader:
            continue

        if reader == "excel":
            sheets = _get_excel_sheets(item)
            if len(sheets) > 1:
                for sheet in sheets:
                    files_by_type[reader].append({
                        "name": f"{_sanitize(item.stem)}_{_sanitize(sheet)}",
                        "path": item.name,
                        "sheet": sheet,
                    })
            else:
                files_by_type[reader].append({
                    "name": _sanitize(item.stem),
                    "path": item.name,
                    "sheet": sheets[0] if sheets else None,
                })
        else:
            files_by_type[reader].append({
                "name": _sanitize(item.stem),
                "path": item.name,
            })

    return files_by_type


def _generate_load_code(data_dir: str, files_by_type: dict[str, list[dict]]) -> str:
    """Build the import + load statements for a notebook cell."""
    lines = [
        "import pandas as pd",
        "from pathlib import Path",
        "",
        f'data_dir = Path("{data_dir}")',
    ]

    for reader_type, files in sorted(files_by_type.items()):
        label = TYPE_LABELS.get(reader_type, f"{reader_type.title()} files")
        lines += ["", f"# {label}"]
        for f in files:
            if reader_type == "excel" and f.get("sheet"):
                lines.append(
                    f'{f["name"]} = pd.read_excel(data_dir / "{f["path"]}", '
                    f'sheet_name="{f["sheet"]}")'
                )
            else:
                lines.append(
                    f'{f["name"]} = pd.read_{reader_type}(data_dir / "{f["path"]}")'
                )

    return "\n".join(lines)


def scan(url: str, notebook_path: str) -> None:
    """Scan a directory for data files and generate a notebook with load commands."""
    directory = Path(url)
    if not directory.is_dir():
        raise ValueError(f"Invalid directory: {url}")

    files_by_type = _collect_data_files(directory)
    code = _generate_load_code(url, files_by_type)

    create_notebook(notebook_path)
    insert_cell(notebook_path, code, cell_type="code")


# ═══════════════════════════════════════════════════════════════════════════════
# Profiling
# ═══════════════════════════════════════════════════════════════════════════════


def _generate_profile_code() -> str:
    """Generate generic profiling code for a DataFrame."""
    return dedent_block("""
        import pandas as pd

        def _profile_print_section(title):
            line = "=" * 70
            print(f"\\n{line}\\n{title}\\n{line}")

        def analyze_profiling(df, name="DataFrame"):
            _profile_print_section(f"Profiling: {name}")

            # Shape
            print(f"\\nShape: {df.shape[0]:,} rows, {df.shape[1]:,} columns")

            # Info
            print("\\n--- Info ---")
            df.info()

            # Head
            print("\\n--- Head (5 rows) ---")
            print(df.head())

            # Tail
            print("\\n--- Tail (5 rows) ---")
            print(df.tail())

            # Describe
            print("\\n--- Describe ---")
            print(df.describe(include="all"))

            # Nulls
            print("\\n--- Null Values ---")
            null_counts = df.isnull().sum()
            null_pct = (df.isnull().sum() / len(df)) * 100
            null_df = pd.DataFrame({"Count": null_counts, "Percentage": null_pct})
            null_df = null_df[null_df["Count"] > 0].sort_values("Count", ascending=False)
            if len(null_df) > 0:
                print(null_df)
            else:
                print("[INFO] No null values found")

            # Uniques
            print("\\n--- Unique Values per Column ---")
            unique_counts = df.nunique().sort_values()
            print(pd.DataFrame({"Unique": unique_counts}))
    """)


def profile_dataframe(notebook_path: str, df_name: str | None = None) -> None:
    """Add DataFrame profiling cells to a notebook."""
    df_names = resolve_df_names(notebook_path, df_name)
    insert_analysis_cells(
        notebook_path,
        df_names,
        _generate_profile_code,
        "profiling",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Type Inference
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_info_output(output_text: str) -> dict:
    """Parse column names and dtypes from df.info() text output."""
    columns = {}
    lines = output_text.split("\n")

    in_columns = False
    for line in lines:
        if "Data columns" in line:
            in_columns = True
            continue
        if in_columns and "---" in line:
            continue
        if in_columns:
            if not line.strip():
                break

            match = re.match(r"\s*\d+\s+(.*?)\s+\d+\s+non-null\s+(\S+)", line)
            if match:
                col_name = match.group(1)
                dtype = match.group(2)
                columns[col_name] = dtype

    return columns


def _parse_unique_output(output_text: str) -> dict:
    """Parse unique counts output from print(df)."""
    uniques = {}
    lines = output_text.split("\n")

    for line in lines:
        line = line.strip()
        if not line or "Unique" in line:
            continue

        parts = line.split()
        if len(parts) >= 2:
            try:
                count = int(parts[-1])
                col_name = " ".join(parts[:-1])
                uniques[col_name] = count
            except ValueError:
                pass

    return uniques


def _extract_dataframe_metadata(nb: dict) -> dict:
    """Extract DataFrame metadata from profiling cell outputs."""
    metadata = defaultdict(lambda: {"columns": {}, "uniques": {}, "row_count": 0})

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        full_output = ""
        for output in cell.get("outputs", []):
            if "text" in output:
                full_output += "".join(output["text"])

        if not full_output:
            continue

        # ── New pattern: call cell like analyze_profiling(df, name='df') ──
        if "analyze_profiling(" in source and "Profiling:" in full_output:
            name_match = re.search(r"name=['\"](.+?)['\"]", source)
            if name_match:
                df_name = name_match.group(1)
            else:
                # Fallback: first positional argument
                arg_match = re.search(r"analyze_profiling\(([^,)]+)", source)
                df_name = arg_match.group(1).strip() if arg_match else None

            if df_name:
                _parse_profile_output(metadata, df_name, full_output)
            continue

        # ── Old pattern: single cell with # Profile: df_name ──
        if "# Profile:" in source:
            df_name = None
            for line in source.split("\n"):
                if "# Profile:" in line:
                    df_name = line.split("# Profile:")[1].strip()
                    break
            if df_name and "Shape:" in full_output:
                _parse_profile_output(metadata, df_name, full_output)

    return dict(metadata)


def _parse_profile_output(metadata: dict, df_name: str, full_output: str) -> None:
    """Parse a single profiling output block into metadata."""
    shape_match = re.search(r"Shape: ([\d,]+) rows", full_output)
    if shape_match:
        row_count = int(shape_match.group(1).replace(",", ""))
        metadata[df_name]["row_count"] = row_count

    if "--- Info ---" in full_output:
        info_section = full_output.split("--- Info ---")[1].split("--- Head")[0]
        columns = _parse_info_output(info_section)
        if "Column" in columns:
            del columns["Column"]
        metadata[df_name]["columns"] = columns

    if "--- Unique Values per Column ---" in full_output:
        unique_section = full_output.split("--- Unique Values per Column ---")[1]
        uniques = _parse_unique_output(unique_section)
        metadata[df_name]["uniques"] = uniques


def _infer_from_metadata(metadata: dict) -> tuple:
    """Infer semantic types from DataFrame metadata."""
    findings = defaultdict(lambda: defaultdict(dict))
    column_names_across_dfs = defaultdict(list)

    def normalize_column_name(name):
        name = re.sub(r"[-\s]+", "_", name)
        name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name)
        name = name.lower()
        name = re.sub(r"_+", "_", name)
        name = name.strip("_")
        return name

    for df_name, df_meta in metadata.items():
        columns = df_meta.get("columns", {})
        uniques = df_meta.get("uniques", {})
        row_count = df_meta.get("row_count", 0)

        for col, dtype in columns.items():
            normalized = normalize_column_name(col)
            column_names_across_dfs[normalized].append((df_name, col))

            date_patterns = ["date", "time", "created", "updated", "modified", "timestamp", "dt", "day"]
            if dtype == "object" and any(p in col.lower() for p in date_patterns):
                findings[df_name][col] = {
                    "issue": "date_as_string",
                    "current_type": dtype,
                    "suggested_type": "datetime64",
                }

            elif dtype == "object":
                numeric_patterns = ["amount", "price", "cost", "value", "total", "count", "number", "qty", "quantity"]
                if any(p in col.lower() for p in numeric_patterns):
                    findings[df_name][col] = {
                        "issue": "numeric_as_string",
                        "current_type": dtype,
                        "suggested_type": "numeric",
                    }

            if row_count > 0:
                unique_count = uniques.get(col, 0)
                unique_ratio = unique_count / row_count if row_count > 0 else 0

                if unique_ratio < 0.05 and unique_count < 50 and dtype == "object":
                    findings[df_name][col] = {
                        "issue": "should_be_categorical",
                        "current_type": dtype,
                        "suggested_type": "category",
                    }

                elif unique_count == 2 and dtype == "object":
                    findings[df_name][col] = {
                        "issue": "boolean_as_string",
                        "current_type": dtype,
                        "suggested_type": "bool",
                    }

                id_patterns = ["id", "key", "code", "uuid", "guid"]
                if unique_ratio > 0.9 and any(p in col.lower() for p in id_patterns):
                    findings[df_name][col] = {
                        "issue": "identifier_column",
                        "current_type": dtype,
                        "suggested_type": "string (keep as-is)",
                    }

                # Whitespace detection: flag all object columns as potentially having whitespace
                if dtype == "object":
                    # Only add if not already flagged with a higher-priority issue
                    if col not in findings[df_name]:
                        findings[df_name][col] = {
                            "issue": "potential_whitespace",
                            "current_type": dtype,
                            "suggested_type": "string (trimmed)",
                        }

    normalization_needed = {}
    for normalized_name, occurrences in column_names_across_dfs.items():
        if len(occurrences) > 1:
            for df_name, original_col in occurrences:
                if original_col != normalized_name:
                    if df_name not in normalization_needed:
                        normalization_needed[df_name] = {}
                    normalization_needed[df_name][original_col] = normalized_name

    return dict(findings), normalization_needed, dict(column_names_across_dfs)


def print_semantic_report(findings: dict, normalization: dict, column_mapping: dict) -> None:
    """Print formatted console report of semantic type analysis."""
    print("\n" + "=" * 80)
    print("SEMANTIC TYPE ANALYSIS REPORT")
    print("=" * 80)

    if findings:
        print("\nTYPE CONVERSION RECOMMENDATIONS")
        print("-" * 80)
        for df_name in sorted(findings.keys()):
            print(f"\nDataFrame: {df_name}")
            for col, info in findings[df_name].items():
                issue = info["issue"].replace("_", " ").title()
                print(f"  - {col}: {issue}")
                print(f"    Current: {info['current_type']} -> Suggested: {info['suggested_type']}")
    else:
        print("\n[INFO] No type conversion issues detected")

    if normalization:
        print("\nCOLUMN NAME NORMALIZATION RECOMMENDATIONS")
        print("-" * 80)
        for df_name in sorted(normalization.keys()):
            print(f"\nDataFrame: {df_name}")
            for original, normalized in normalization[df_name].items():
                print(f"  - {original} -> {normalized}")
    else:
        print("\n[INFO] Column names are already normalized")

    # Whitespace findings
    whitespace_findings = {}
    for df_name, cols in findings.items():
        ws_cols = {c: info for c, info in cols.items() if info["issue"] == "potential_whitespace"}
        if ws_cols:
            whitespace_findings[df_name] = ws_cols

    if whitespace_findings:
        print("\nWHITESPACE DETECTION")
        print("-" * 80)
        for df_name in sorted(whitespace_findings.keys()):
            print(f"\nDataFrame: {df_name}")
            for col in sorted(whitespace_findings[df_name].keys()):
                print(f"  - {col}: potential leading/trailing whitespace")
    else:
        print("\n[INFO] No object columns to check for whitespace")

    shared_cols = {norm: occs for norm, occs in column_mapping.items() if len(occs) > 1}
    if shared_cols:
        print("\nSHARED COLUMNS ACROSS DATAFRAMES")
        print("-" * 80)
        for normalized_name, occurrences in sorted(shared_cols.items()):
            df_list = [df_name for df_name, _ in occurrences]
            print(f"  - {normalized_name}: present in {', '.join(df_list)}")

    print("\n" + "=" * 80 + "\n")


def infer_semantic_types(notebook_path: str, df_name: str | None = None) -> dict:
    """Analyze DataFrames for semantic type issues.

    Args:
        notebook_path: Path to the notebook file.
        df_name: Optional explicit DataFrame name to analyze. If None, analyzes all.

    Returns:
        dict with keys:
            - findings: Type conversion issues per DataFrame/column
            - normalization: Column renaming mappings
            - column_mapping: Shared columns across DataFrames
            - metadata: Row counts, dtypes, unique counts
    """
    nb = read_notebook(notebook_path)
    metadata = _extract_dataframe_metadata(nb)

    if not metadata:
        raise ValueError("No profiling outputs found. Run profile_dataframe() first and execute the cells.")

    if df_name is not None and df_name not in metadata:
        raise ValueError(f"DataFrame '{df_name}' not found in profiling outputs.")

    if df_name is not None:
        metadata = {df_name: metadata[df_name]}

    findings, normalization_needed, column_mapping = _infer_from_metadata(metadata)

    return {
        "findings": findings,
        "normalization": normalization_needed,
        "column_mapping": column_mapping,
        "metadata": metadata,
    }
