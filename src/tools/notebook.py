"""
Notebook Handler
Loads and parses .ipynb files, wraps nb-cli for cell execution,
inserts cells with generated code/results.
"""

import ast
import json
import re
import shutil
import subprocess
import uuid
import nbformat
from pathlib import Path
from datetime import datetime
from typing import Dict, Any                                                                                          
try:
    import nbformat
    NBFORMAT_AVAILABLE = True
except ImportError:
    NBFORMAT_AVAILABLE = False


def require_nb():
    """Check nb CLI (Jupyter) is available."""
    # Check common install locations
    nb_paths = [
        Path.home() / ".nb-cli" / "bin" / "nb",
        Path.home() / ".local" / "bin" / "nb",
        Path("/usr/local/bin/nb"),
    ]
    
    for nb_path in nb_paths:
        if nb_path.exists():
            return
    
    raise RuntimeError(
        "nb CLI (Jupyter) not found. Install: curl -fsSL "
        "https://raw.githubusercontent.com/jupyter-ai-contrib/nb-cli/main/install.sh | bash"
    )


def get_nb_path() -> Path:
    """Get path to nb CLI."""
    require_nb()
    # Return first found path
    nb_paths = [
        Path.home() / ".nb-cli" / "bin" / "nb",
        Path.home() / ".local" / "bin" / "nb",
        Path("/usr/local/bin/nb"),
    ]
    for nb_path in nb_paths:
        if nb_path.exists():
            return nb_path
    return Path("nb")

def upgrade_notebook(path):
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # ensure latest structure
    nb = nbformat.v4.upgrade(nb)

    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

def read_notebook(path: str) -> dict:
    """Load .ipynb JSON file."""
    require_nb()
    nb_cmd = str(get_nb_path())
    result = subprocess.run(
        [nb_cmd, "read", path, "--json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        try:
            upgrade_notebook(path)
            return read_notebook(path)
        except Exception:
            raise RuntimeError(f"Failed to read notebook: {result.stderr}")
    return json.loads(result.stdout)

def create_notebook(path: str, force: bool = False) -> dict:
    """Create a blank Jupyter notebook."""
    require_nb()
    nb_cmd = str(get_nb_path())
    
    cmd = [nb_cmd, "create", path]
    if force:
        cmd.append("--force")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create notebook: {result.stderr}")
    
    return read_notebook(path)

def save_notebook(nb: dict, path: str, create_backup: bool = False) -> None:
    """Save notebook dict to .ipynb file with optional backup."""
    if not isinstance(nb, dict):
        raise TypeError(f"nb must be dict, got {type(nb).__name__}")

    path = Path(path)
    if path.suffix != '.ipynb':
        raise ValueError(f"Expected .ipynb file, got {path.suffix}")

    # Create timestamped backup before overwriting
    if create_backup and path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.stem}.backup_{timestamp}.ipynb")
        shutil.copy2(path, backup_path)
        print(f"Backup created: {backup_path}")

    # Use nbformat for proper validation and writing
    try:
        nb_obj = nbformat.from_dict(nb)
        nbformat.validate(nb_obj)

        with open(path, 'w', encoding='utf-8') as f:
            nbformat.write(nb_obj, f)
    except Exception as e:
        # If nbformat fails, fallback to JSON but warn
        print(f"Warning: nbformat write failed ({e}), using JSON fallback")
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(nb, f, indent=1)
        except PermissionError:
            raise RuntimeError(f"Permission denied: {path}")
        except FileNotFoundError:
            raise RuntimeError(f"Directory not found: {path.parent}")

def insert_cell(
      url: str = None,
      code: str = None,
      nb: dict = None,
      position: int = -1,
      cell_type: str = "code",
  ) -> dict:

      if not code:
          raise ValueError("code is required")
      if cell_type not in ("code", "markdown"):
          raise ValueError(f"cell_type must be 'code' or 'markdown', got {cell_type}")
      if url is None and nb is None:
          raise ValueError("Must provide either url or nb")
      if url is not None and nb is not None:
          raise ValueError("Provide only one of url or nb, not both")

      if url:
          nb = read_notebook(url)

      # Generate unique cell ID
      cell_id = str(uuid.uuid4()).replace('-', '')[:8]

      if cell_type == "code":
          new_cell = {
              "cell_type": "code",
              "id": cell_id,  # Add unique ID
              "execution_count": None,
              "metadata": {},
              "source": [code + "\n"],
              "outputs": []
          }
      else:
          new_cell = {
              "cell_type": "markdown",
              "id": cell_id,  # Add unique ID
              "metadata": {},
              "source": [code + "\n"]
          }

      insert_pos = position if position != -1 else len(nb["cells"])
      nb["cells"].insert(insert_pos, new_cell)

      if url:
          save_notebook(nb, url)
      else:
          return nb


def insert_cells_batch(notebook_path: str, cells: list[dict]) -> None:
    """Insert multiple cells at once - SAFER than repeated insert_cell calls.

    Args:
        notebook_path: Path to the notebook
        cells: List of dicts with keys:
            - "code": The cell source code
            - "type": "code" or "markdown"
            - "position": (optional) where to insert, default appends to end
    """
    nb = read_notebook(notebook_path)

    for cell_data in cells:
        cell_id = str(uuid.uuid4()).replace('-', '')[:8]
        cell_type = cell_data.get("type", "code")
        code = cell_data["code"]
        position = cell_data.get("position", -1)

        if cell_type == "code":
            new_cell = {
                "cell_type": "code",
                "id": cell_id,
                "execution_count": None,
                "metadata": {},
                "source": [code + "\n"],
                "outputs": []
            }
        else:
            new_cell = {
                "cell_type": "markdown",
                "id": cell_id,
                "metadata": {},
                "source": [code + "\n"]
            }

        insert_pos = position if position != -1 else len(nb["cells"])
        nb["cells"].insert(insert_pos, new_cell)

    # Single save at the end - much safer!
    save_notebook(nb, notebook_path)


def execute_cell(nb_path, cell_index):
    require_nb()
    nb_cmd = str(get_nb_path())
    result = subprocess.run(
        [nb_cmd, "execute", "--start", "0", "--end", str(cell_index), nb_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to execute: {result.stderr}")
    

def execute_notebook(path: str, timeout: int = 300) -> None:
      """Execute all cells in notebook."""
      require_nb()
      nb_cmd = str(get_nb_path())
      result = subprocess.run(
          [nb_cmd, "execute", "--timeout", str(timeout), path],
          capture_output=True,
          text=True,
      )
      if result.returncode != 0:
          raise RuntimeError(f"Failed to execute notebook: {result.stderr}")


def get_cell_output(nb_path: str, cell_index: int) -> str:
    """Extract output from executed cell."""
    nb = read_notebook(nb_path)
    cells = nb.get("cells", [])

    if cell_index >= len(cells):
        raise IndexError(f"Cell index {cell_index} is out of range for notebook with {len(cells)} cells.")

    cell = cells[cell_index]
    cell_type = cell.get("cell_type")

    if cell_type == "markdown":
        source = cell.get("source", "")
        return "".join(source) if isinstance(source, list) else source

    outputs = cell.get("outputs", [])
    if not outputs:
        return None

    output = outputs[0]
    output_type = output.get("output_type")

    if output_type == "stream":
        return "".join(output.get("text", []))
    elif output_type in ("execute_result", "display_data"):  # Added display_data
        data = output.get("data", {})
        # Prefer plain text, fallback to HTML
        return data.get("text/plain") or data.get("text/html")
    elif output_type == "error":
        return "\n".join(output.get("traceback", []))

    return None


def find_dataframe_cells(nb: dict) -> list[tuple[int, str]]:
    """Find cells with DataFrame definitions using AST parsing."""
    results = []

    for i, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        source = "".join(cell.get("source", []))
        if not source.strip():
            continue

        try:
            tree = ast.parse(source)
            df_names = _extract_dataframe_names(tree)
            for name in df_names:
                results.append((i, name))
        except SyntaxError:
            # If AST parsing fails, fall back to pattern matching
            df_names = _pattern_match_dataframes(source)
            for name in df_names:
                results.append((i, name))

    return results


def _extract_dataframe_names(tree: ast.AST) -> list[str]:
    """Extract variable names that are likely DataFrames from AST."""
    df_names = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if _is_pandas_read_call(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        df_names.append(target.id)
            elif _is_dataframe_constructor(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        df_names.append(target.id)
            elif _is_dataframe_operation(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        df_names.append(target.id)

    return df_names


def _is_pandas_read_call(node: ast.AST) -> bool:
    """Check if node is a pandas read_* call."""
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name):
            if node.func.value.id == "pd" and node.func.attr.startswith("read_"):
                return True
    return False


def _is_dataframe_constructor(node: ast.AST) -> bool:
    """Check if node is a DataFrame constructor call."""
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name):
            if node.func.value.id in ("pd", "pandas") and node.func.attr == "DataFrame":
                return True
    return False


def _is_dataframe_operation(node: ast.AST) -> bool:
    """Check if node is a DataFrame operation that returns a DataFrame."""
    if not isinstance(node, ast.Call):
        return False

    df_methods = {
        'merge', 'join', 'concat', 'groupby', 'pivot', 'pivot_table',
        'drop', 'dropna', 'fillna', 'reset_index', 'set_index',
        'sort_values', 'sort_index', 'query', 'copy', 'sample',
    }

    if isinstance(node.func, ast.Attribute):
        if node.func.attr in df_methods:
            return True
        if isinstance(node.func.value, ast.Name):
            if node.func.value.id == "pd" and node.func.attr in {'concat', 'merge'}:
                return True
    return False


def _pattern_match_dataframes(source: str) -> list[str]:
    """Fallback: use regex patterns to find likely DataFrame assignments."""
    patterns = [
        r'(\w+)\s*=\s*pd\.read_\w+\(',
        r'(\w+)\s*=\s*pd\.DataFrame\(',
        r'(\w+)\s*=\s*pandas\.read_\w+\(',
        r'(\w+)\s*=\s*pandas\.DataFrame\(',
    ]

    names = []
    for pattern in patterns:
        matches = re.findall(pattern, source)
        names.extend(matches)

    return list(set(names))





