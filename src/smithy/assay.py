"""
Structural Mapping

- Scans notebook for DataFrames → profiles (head/tail/info/describe/nulls/uniques/shapes)
- Semantic validation (dates → pd.to_datetime, numbers → pd.to_numeric, sense-checks)
- Inserts summary table + suggested fixes cell.
"""

from tools.notebook import create_notebook, insert_cell
from pathlib import Path
from collections import defaultdict
import openpyxl

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
                # Single sheet — load normally, no sheet_name needed
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


def _generate_code(data_dir: str, files_by_type: dict[str, list[dict]]) -> str:
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
    code = _generate_code(url, files_by_type)

    create_notebook(notebook_path)
    insert_cell(notebook_path, code, cell_type="code")


def profile_dataframe(): 
    return

def infer_semantic_types(): 
    return

def sense_check(): 
    return

def generate_report(): 
    return