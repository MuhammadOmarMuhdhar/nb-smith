"""
Notebook Handler

Wraps nb-cli (auto-downloaded) for notebook I/O and execution.
Falls back to direct nbformat operations where nb-cli has no equivalent command.
"""

import ast
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

import nbformat

# ── Constants ────────────────────────────────────────────────────────────────
NB_CLI_VERSION = "v0.0.8"
CACHE_DIR = Path.home() / ".nb-smith" / "bin"
CACHED_NB = CACHE_DIR / "nb"

_RELEASE_BASE = (
    "https://github.com/jupyter-ai-contrib/nb-cli/releases/download"
    f"/{NB_CLI_VERSION}"
)

_PLATFORM_MAP = {
    ("darwin", "x86_64"): "nb-macos-amd64",
    ("darwin", "arm64"): "nb-macos-arm64",
    ("linux", "x86_64"): "nb-linux-amd64",
    ("linux", "aarch64"): "nb-linux-arm64",
    ("linux", "arm64"): "nb-linux-arm64",
    ("windows", "amd64"): "nb-windows-amd64.exe",
}


# ═══════════════════════════════════════════════════════════════════════════════
# nb-cli Discovery / Auto-download
# ═══════════════════════════════════════════════════════════════════════════════


def _get_platform_asset() -> str:
    """Return the release asset name for the current platform."""
    system = platform.system().lower()
    machine = platform.machine().lower()
    key = (system, machine)
    if key not in _PLATFORM_MAP:
        raise RuntimeError(
            f"Unsupported platform: {system} {machine}. "
            f"nb-smith requires one of: {list(_PLATFORM_MAP.values())}"
        )
    return _PLATFORM_MAP[key]


def _verify_checksum(binary_path: Path, expected_hash: str) -> bool:
    """SHA-256 verify a downloaded binary."""
    sha256 = hashlib.sha256()
    with open(binary_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest().lower() == expected_hash.lower()


def _download_nb_cli() -> Path:
    """Download the correct nb-cli binary to the cache directory."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    asset = _get_platform_asset()
    binary_url = f"{_RELEASE_BASE}/{asset}"
    sums_url = f"{_RELEASE_BASE}/SHA256SUMS"

    # ── 1. Fetch SHA256SUMS ────────────────────────────────────────────────
    try:
        with urllib.request.urlopen(sums_url, timeout=60) as response:
            sums_content = response.read().decode("utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to fetch SHA256SUMS: {exc}")

    expected_hash = None
    for line in sums_content.strip().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == asset:
            expected_hash = parts[0]
            break

    if not expected_hash:
        raise RuntimeError(f"Checksum for '{asset}' not found in SHA256SUMS")

    # ── 2. Download binary to temp file ────────────────────────────────────
    print(f"Downloading {asset} (this may take a minute)...")
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            # Use longer timeout for slow connections
            with urllib.request.urlopen(binary_url, timeout=300) as response:
                shutil.copyfileobj(response, tmp)
            tmp_path = Path(tmp.name)
    except Exception as exc:
        raise RuntimeError(f"Failed to download nb-cli binary: {exc}")

    # ── 3. Verify ──────────────────────────────────────────────────────────
    if not _verify_checksum(tmp_path, expected_hash):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(
            "Checksum verification failed for downloaded nb-cli binary. "
            "This may indicate a corrupted download or man-in-the-middle attack."
        )

    # ── 4. Move to cache + make executable ─────────────────────────────────
    shutil.move(str(tmp_path), str(CACHED_NB))

    if platform.system() != "Windows":
        CACHED_NB.chmod(CACHED_NB.stat().st_mode | stat.S_IEXEC)

    print(f"nb-cli cached: {CACHED_NB}")
    return CACHED_NB


def _is_notebook_cli(path: Path) -> bool:
    """Verify that the given 'nb' binary is the notebook CLI, not NoneBot."""
    try:
        result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = (result.stdout + result.stderr).lower()
        # NoneBot CLI prints "nonebot cli version X.X.X"
        # Notebook CLI (nb-cli) prints "nb X.X.X" or similar
        if "nonebot" in output:
            return False
        return True
    except Exception:
        return False


def ensure_nb_cli(auto_approve: bool = False) -> Path:
    """
    Locate or download the nb-cli binary.

    Checks (in order):
      1. Cached binary at ~/.nb-smith/bin/nb
      2. 'nb' on PATH (verified it's the notebook CLI, not NoneBot)
      3. Download from GitHub releases (with user consent)

    Args:
        auto_approve: If True, skip the download consent prompt.
                      Also respects NB_SMITH_AUTO_APPROVE=1 env var.

    Returns:
        Path to the nb-cli executable.

    Raises:
        RuntimeError: If nb-cli cannot be found or downloaded.
    """
    auto_approve = auto_approve or os.environ.get("NB_SMITH_AUTO_APPROVE") == "1"

    # 1. Cached binary (preferred)
    if CACHED_NB.exists():
        return CACHED_NB

    # 2. PATH — but verify it's not NoneBot CLI
    nb_from_path = shutil.which("nb")
    if nb_from_path:
        nb_path = Path(nb_from_path)
        if _is_notebook_cli(nb_path):
            return nb_path

    # 3. Prompt & download
    if not auto_approve:
        print("\n" + "=" * 70)
        print("nb-cli (notebook CLI) is required but not found on your system.")
        print(f"It will be downloaded and cached at: {CACHED_NB}")
        print("Source: https://github.com/jupyter-ai-contrib/nb-cli")
        print("=" * 70)
        response = input("Download now? [Y/n]: ").strip().lower()
        if response and response not in ("y", "yes"):
            raise RuntimeError(
                "nb-cli is required. Install manually from "
                "https://github.com/jupyter-ai-contrib/nb-cli "
                "or re-run with --yes / -y to auto-approve."
            )

    return _download_nb_cli()


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook Operations (via nb-cli)
# ═══════════════════════════════════════════════════════════════════════════════


def read_notebook(path: str) -> dict:
    """Load a notebook using nb-cli --json for reliable parsing."""
    nb_cmd = str(ensure_nb_cli())
    result = subprocess.run(
        [nb_cmd, "read", path, "--json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to read notebook: {result.stderr}")
    return json.loads(result.stdout)


def create_notebook(path: str, force: bool = False) -> dict:
    """Create a blank Jupyter notebook using nb-cli."""
    nb_cmd = str(ensure_nb_cli())
    cmd = [nb_cmd, "create", path]
    if force:
        cmd.append("--force")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Failed to create notebook: {result.stderr}")
    return read_notebook(path)


def execute_notebook(path: str, timeout: int = 300) -> None:
    """Execute all cells in a notebook using nb-cli."""
    nb_cmd = str(ensure_nb_cli())
    result = subprocess.run(
        [nb_cmd, "execute", "--timeout", str(timeout), path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to execute notebook: {result.stderr}")


# ═══════════════════════════════════════════════════════════════════════════════
# Notebook Operations (direct nbformat — nb-cli has no equivalent)
# ═══════════════════════════════════════════════════════════════════════════════


def upgrade_notebook(path: str) -> None:
    """Upgrade notebook to latest nbformat version."""
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    nb = nbformat.v4.upgrade(nb)

    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def save_notebook(nb: dict, path: str, create_backup: bool = False) -> None:
    """Save notebook dict to .ipynb file with optional backup."""
    if not isinstance(nb, dict):
        raise TypeError(f"nb must be dict, got {type(nb).__name__}")

    path = Path(path)
    if path.suffix != ".ipynb":
        raise ValueError(f"Expected .ipynb file, got {path.suffix}")

    if create_backup and path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = path.with_name(f"{path.stem}.backup_{timestamp}.ipynb")
        shutil.copy2(path, backup_path)
        print(f"Backup created: {backup_path}")

    nb.setdefault("nbformat", 4)
    nb.setdefault("nbformat_minor", 5)
    nb.setdefault("metadata", {})
    nb.setdefault("cells", [])

    nb["nbformat"] = 4
    nb["nbformat_minor"] = 5

    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            valid_keys = {
                "cell_type", "execution_count", "metadata", "outputs", "source", "id"
            }
        else:
            valid_keys = {"cell_type", "metadata", "source", "id"}

        for key in list(set(cell.keys()) - valid_keys):
            del cell[key]

    try:
        nb_obj = nbformat.from_dict(nb)
        nbformat.validate(nb_obj)
        with open(path, "w", encoding="utf-8") as f:
            nbformat.write(nb_obj, f)
    except Exception as e:
        print(f"Warning: nbformat write failed ({e}), using JSON fallback")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=1)


def insert_cell(
    url: str | None = None,
    code: str | None = None,
    nb: dict | None = None,
    position: int = -1,
    cell_type: str = "code",
) -> dict | None:
    """Insert a single cell into a notebook."""
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

    cell_id = str(uuid.uuid4()).replace("-", "")[:8]

    if cell_type == "code":
        new_cell = {
            "cell_type": "code",
            "id": cell_id,
            "execution_count": None,
            "metadata": {},
            "source": [code + "\n"],
            "outputs": [],
        }
    else:
        new_cell = {
            "cell_type": "markdown",
            "id": cell_id,
            "metadata": {},
            "source": [code + "\n"],
        }

    insert_pos = position if position != -1 else len(nb["cells"])
    nb["cells"].insert(insert_pos, new_cell)

    if url:
        save_notebook(nb, url)
        return None
    return nb


def insert_cells_batch(notebook_path: str, cells: list[dict]) -> None:
    """Insert multiple cells at once — safer than repeated insert_cell calls."""
    nb = read_notebook(notebook_path)

    for cell_data in cells:
        cell_id = str(uuid.uuid4()).replace("-", "")[:8]
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
                "outputs": [],
            }
        else:
            new_cell = {
                "cell_type": "markdown",
                "id": cell_id,
                "metadata": {},
                "source": [code + "\n"],
            }

        insert_pos = position if position != -1 else len(nb["cells"])
        nb["cells"].insert(insert_pos, new_cell)

    save_notebook(nb, notebook_path)


def execute_cell(nb_path: str, cell_index: int) -> None:
    """Execute notebook cells up to a given index."""
    nb_cmd = str(ensure_nb_cli())
    result = subprocess.run(
        [nb_cmd, "execute", "--start", "0", "--end", str(cell_index), nb_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Failed to execute: {result.stderr}")


def get_cell_output(nb_path: str, cell_index: int) -> str | None:
    """Extract output from an executed cell."""
    nb = read_notebook(nb_path)
    cells = nb.get("cells", [])

    if cell_index >= len(cells):
        raise IndexError(
            f"Cell index {cell_index} is out of range for notebook with {len(cells)} cells."
        )

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
    elif output_type in ("execute_result", "display_data"):
        data = output.get("data", {})
        return data.get("text/plain") or data.get("text/html")
    elif output_type == "error":
        return "\n".join(output.get("traceback", []))

    return None


# ═══════════════════════════════════════════════════════════════════════════════
# DataFrame Discovery
# ═══════════════════════════════════════════════════════════════════════════════


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
            df_names = _pattern_match_dataframes(source)
            for name in df_names:
                results.append((i, name))

    return results


def _extract_dataframe_names(tree: ast.AST) -> set[str]:
    """Extract variable names that are likely DataFrames from AST.

    Only considers top-level (module-level) assignments.
    Ignores assignments inside functions, classes, conditionals, loops, etc.
    """
    df_names: set[str] = set()

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            if _is_pandas_read_call(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        df_names.add(target.id)
            elif _is_dataframe_constructor(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        df_names.add(target.id)
            elif _is_dataframe_operation(node.value):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        df_names.add(target.id)

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
        "merge", "join", "concat", "groupby", "pivot", "pivot_table",
        "drop", "dropna", "fillna", "reset_index", "set_index",
        "sort_values", "sort_index", "query", "copy", "sample",
    }

    if isinstance(node.func, ast.Attribute):
        if node.func.attr in df_methods:
            return True
        if isinstance(node.func.value, ast.Name):
            if node.func.value.id == "pd" and node.func.attr in {"concat", "merge"}:
                return True
    return False


def _pattern_match_dataframes(source: str) -> set[str]:
    """Fallback: use regex patterns to find likely DataFrame assignments.

    Only matches at the start of a line (module-level only).
    Ignores indented code inside functions, loops, conditionals, etc.
    """
    patterns = [
        r"^(\w+)\s*=\s*pd\.read_\w+\(",
        r"^(\w+)\s*=\s*pd\.DataFrame\(",
        r"^(\w+)\s*=\s*pandas\.read_\w+\(",
        r"^(\w+)\s*=\s*pandas\.DataFrame\(",
    ]

    names: set[str] = set()
    for pattern in patterns:
        # re.MULTILINE makes ^ match after each newline
        matches = re.findall(pattern, source, re.MULTILINE)
        names.update(matches)

    return names
