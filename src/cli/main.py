"""
CLI Entry Point
Typer-based CLI for nb-smith.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import typer

from smithy import assay as assay_module
from smithy import cupel as cupel_module
from smithy import smelt as smelt_module
from smithy.cast import build_pipeline as cast_build_pipeline
from tools.notebook import execute_notebook

# ── Rich help panels ──────────────────────────────────────────────────────────
DISCOVERY_PANEL = "Discovery"
ANALYSIS_PANEL = "Analysis"
DIAGNOSTICS_PANEL = "Diagnostics"
ALL_IN_ONE_PANEL = "All-in-one"
UTILITY_PANEL = "Utility"

app = typer.Typer()
assay_app = typer.Typer(rich_help_panel=DISCOVERY_PANEL)
smelt_app = typer.Typer(rich_help_panel=ANALYSIS_PANEL)
cupel_app = typer.Typer(rich_help_panel=DIAGNOSTICS_PANEL)


# ═══════════════════════════════════════════════════════════════════════════════
# Help text generators
# ═══════════════════════════════════════════════════════════════════════════════


def _print_examples() -> None:
    """Print usage examples with Rich nested panels."""
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.text import Text
    from rich import box

    console = Console()

    def _colorize(cmd: str) -> Text:
        text = Text()
        parts = cmd.split()
        if not parts:
            return text
        text.append(parts[0], style="bold green")
        subcommands = {
            "assay", "smelt", "cupel", "forge", "execute",
            "scan", "profile", "infer",
            "dates", "numerics", "booleans", "categoricals", "normalize", "trim",
            "missingness", "outliers", "duplicates", "cardinality", "skew",
        }
        for part in parts[1:]:
            if part.startswith("--") or part.startswith("-"):
                text.append(f" {part}", style="bold yellow")
            elif part in subcommands:
                text.append(f" {part}", style="bold cyan")
            else:
                text.append(f" {part}", style="dim")
        return text

    def _example_panel(title: str, *commands: str) -> Panel:
        lines = Group(*[_colorize(cmd) for cmd in commands])
        return Panel(
            lines,
            title=Text(title, style="bold bright_cyan"),
            title_align="left",
            border_style="blue",
            padding=(0, 1),
            box=box.ROUNDED,
        )

    content = Group(
        _example_panel("Quick start (one command)",
            "nb-smith forge --from ./data --output analysis.ipynb"),
        _example_panel("Step by step",
            "nb-smith assay scan ./data --output analysis.ipynb",
            "nb-smith assay profile analysis.ipynb",
            "nb-smith execute analysis.ipynb",
            "nb-smith assay infer analysis.ipynb",
            "nb-smith smelt --notebook analysis.ipynb",
            "nb-smith cupel --notebook analysis.ipynb"),
        _example_panel("Target one DataFrame",
            "nb-smith assay profile analysis.ipynb --df customers",
            "nb-smith assay infer analysis.ipynb --df customers --json",
            "nb-smith smelt --notebook analysis.ipynb --df customers",
            "nb-smith cupel --notebook analysis.ipynb --df customers"),
        _example_panel("Granular smelt",
            "nb-smith smelt dates analysis.ipynb --df orders",
            "nb-smith smelt numerics analysis.ipynb --cols amount,price",
            "nb-smith smelt booleans analysis.ipynb --cols is_active",
            "nb-smith smelt categoricals analysis.ipynb --cols status",
            "nb-smith smelt normalize analysis.ipynb --df products",
            "nb-smith smelt trim analysis.ipynb --df products --cols name"),
        _example_panel("Individual diagnostics",
            "nb-smith cupel missingness analysis.ipynb --df orders",
            "nb-smith cupel outliers analysis.ipynb",
            "nb-smith cupel duplicates analysis.ipynb --df customers",
            "nb-smith cupel cardinality analysis.ipynb",
            "nb-smith cupel skew analysis.ipynb --df sales"),
        _example_panel("Automation / CI",
            "nb-smith --yes forge --from ./data --output analysis.ipynb",
            "nb-smith --yes execute analysis.ipynb --timeout 120"),
        _example_panel("JSON output for scripting",
            "nb-smith assay infer analysis.ipynb --json > findings.json"),
    )

    console.print(Panel(
        content,
        title="Examples",
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
        box=box.ROUNDED,
    ))


def _print_functions() -> None:
    """Print functions reference with Rich nested panels."""
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.text import Text
    from rich import box

    console = Console()

    def _cmd(signature: str, description: str) -> Panel:
        return Panel(
            Text(description, style="default"),
            title=Text(signature, style="bold cyan"),
            title_align="left",
            border_style="blue",
            padding=(0, 1),
            box=box.ROUNDED,
        )

    def _section(name: str, *panels: Panel) -> Group:
        header = Text(name, style="bold bright_cyan")
        return Group(header, *panels)

    assay_section = _section(
        "assay",
        _cmd("scan <data-dir> --output <notebook>",
             "Scan directory for CSV, Excel, JSON, Parquet, etc.\nCreate notebook with pd.read_*() load cells."),
        _cmd("profile <notebook> [--df <name>]",
             "Insert profiling helper + per-DataFrame call cells.\nOutputs: shape, dtypes, head, tail, describe, nulls,\nuniques. Must execute cells before running infer."),
        _cmd("infer <notebook> [--df <name>] [--json]",
             "Analyze executed profiling outputs. Detects:\ndate strings, numeric strings, booleans, categoricals,\nwhitespace, identifier columns."),
    )

    smelt_section = _section(
        "smelt",
        _cmd("(no subcommand) --notebook <notebook> [--df <name>]",
             "Full smelt: auto-detect issues and apply all fixes."),
        _cmd("dates <notebook> [--df <name>] [--cols <c1,c2>]",
             "Insert pd.to_datetime() coercion cells."),
        _cmd("numerics <notebook> [--df <name>] [--cols <c1,c2>]",
             "Insert pd.to_numeric() coercion cells."),
        _cmd("booleans <notebook> [--df <name>] [--cols <c1,c2>]",
             "Insert boolean mapping cells."),
        _cmd("categoricals <notebook> [--df <name>] [--cols <c1,c2>]",
             "Insert astype('category') cells."),
        _cmd("normalize <notebook> [--df <name>]",
             "Insert df.rename() for snake_case column names."),
        _cmd("trim <notebook> [--df <name>] [--cols <c1,c2>]",
             "Insert .str.strip() cells for whitespace removal."),
    )

    cupel_section = _section(
        "cupel",
        _cmd("(no subcommand) --notebook <notebook> [--df <name>]",
             "Insert all 5 diagnostic suites."),
        _cmd("missingness <notebook> [--df <name>]",
             "MCAR test, MAR detection, MNAR candidate flagging."),
        _cmd("outliers <notebook> [--df <name>]",
             "IQR (1.5x / 3.0x) and Z-score (>3) outlier detection."),
        _cmd("duplicates <notebook> [--df <name>]",
             "Duplicate row detection with statistical association\ntesting."),
        _cmd("cardinality <notebook> [--df <name>]",
             "Distinct value counts and percentage per column."),
        _cmd("skew <notebook> [--df <name>]",
             "Skewness analysis for numeric columns."),
    )

    pipeline_section = _section(
        "pipeline",
        _cmd("forge --from <data-dir> --output <notebook> [--df <name>]",
             "Full pipeline: scan, profile, execute, assay, smelt,\ncupel, execute. One command from raw data to fully\ndiagnosed notebook."),
        _cmd("execute <notebook> [--timeout <seconds>]",
             "Execute all cells via nb-cli. Required between phases\nthat parse prior cell outputs."),
    )

    content = Group(
        assay_section, Text(""),
        smelt_section, Text(""),
        cupel_section, Text(""),
        pipeline_section,
    )

    console.print(Panel(
        content,
        title="Functions Reference",
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
        box=box.ROUNDED,
    ))


# ═══════════════════════════════════════════════════════════════════════════════
# Global callback
# ═══════════════════════════════════════════════════════════════════════════════


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve nb-cli download"),
    examples: bool = typer.Option(False, "--examples", help="Show usage examples and exit"),
    functions: bool = typer.Option(False, "--functions", help="Show functions reference and exit"),
) -> None:
    """nb-smith is an opinionated CLI tool for automating the early work of
    data analysis. It scans a directory of data files, creates a Jupyter
    notebook, and generates code cells to load, profile, fix types, and
    diagnose data quality."""
    if yes:
        os.environ["NB_SMITH_AUTO_APPROVE"] = "1"
    if examples:
        _print_examples()
        raise typer.Exit()
    if functions:
        _print_functions()
        raise typer.Exit()
    if ctx.invoked_subcommand is None and not yes:
        typer.echo(ctx.get_help())
        raise typer.Exit()


# ═══════════════════════════════════════════════════════════════════════════════
# ASSAY
# ═══════════════════════════════════════════════════════════════════════════════


@assay_app.callback(invoke_without_command=True)
def assay_full(
    ctx: typer.Context,
    from_dir: Optional[Path] = typer.Option(None, "--from", help="Data directory to scan"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output notebook path"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
) -> None:
    """Phase 1: Structural mapping. Scan data directory, create notebook with
    load cells, profile DataFrames, and infer semantic types (dates, numerics,
    booleans, categoricals, whitespace)."""
    if ctx.invoked_subcommand is not None:
        return

    if not from_dir or not output:
        typer.echo("Error: --from and --output are required for full assay run.")
        raise typer.Exit(1)

    typer.echo(f"[assay] Scanning {from_dir} ...")
    assay_module.scan(str(from_dir), str(output))

    typer.echo("[assay] Profiling DataFrames ...")
    assay_module.profile_dataframe(str(output), df_name=df)

    typer.echo("[assay] Executing notebook ...")
    execute_notebook(str(output))

    typer.echo("[assay] Running semantic type inference ...")
    result = assay_module.infer_semantic_types(str(output), df_name=df)

    typer.echo("[assay] Report:")
    assay_module.print_semantic_report(
        result["findings"],
        result["normalization"],
        result["column_mapping"],
    )


@assay_app.command()
def scan(
    data_dir: Path = typer.Argument(..., help="Directory containing data files"),
    output: Path = typer.Option(..., "--output", "-o", help="Output notebook path"),
) -> None:
    """Scan directory for data files and create a notebook with load commands."""
    assay_module.scan(str(data_dir), str(output))
    typer.echo(f"Created notebook: {output}")


@assay_app.command()
def profile(
    notebook: Path = typer.Argument(..., help="Target notebook (.ipynb)"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
) -> None:
    """Insert DataFrame profiling cells. Outputs shape, dtypes, head, tail,
    describe, nulls, and uniques. Must execute cells before running infer."""
    assay_module.profile_dataframe(str(notebook), df_name=df)


@assay_app.command()
def infer(
    notebook: Path = typer.Argument(..., help="Target notebook (.ipynb)"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
    json_output: bool = typer.Option(False, "--json", help="Output raw findings as JSON"),
) -> None:
    """Analyze executed profiling outputs and detect type issues: date strings,
    numeric strings, booleans, categoricals, whitespace, identifier columns."""
    try:
        result = assay_module.infer_semantic_types(str(notebook), df_name=df)
    except ValueError as e:
        typer.echo(f"Error: {e}")
        typer.echo("Hint: Run 'nb-smith assay profile' first, execute cells, then retry.")
        raise typer.Exit(1)

    if json_output:
        typer.echo(json.dumps(result, indent=2, default=str))
    else:
        assay_module.print_semantic_report(
            result["findings"],
            result["normalization"],
            result["column_mapping"],
        )


app.add_typer(assay_app, name="assay")

# ═══════════════════════════════════════════════════════════════════════════════
# SMELT
# ═══════════════════════════════════════════════════════════════════════════════


def _parse_cols(cols: Optional[str]) -> Optional[list[str]]:
    """Parse comma-separated column names."""
    if not cols:
        return None
    return [c.strip() for c in cols.split(",")]


@smelt_app.callback(invoke_without_command=True)
def smelt_full(
    ctx: typer.Context,
    notebook: Optional[Path] = typer.Option(None, "--notebook", "-n", help="Target notebook (.ipynb)"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
) -> None:
    """Phase 2: Type fixes. Apply safe coercions and normalization based on
    assay findings. Inserts code cells for dates, numerics, booleans,
    categoricals, column renaming, and whitespace trimming."""
    if ctx.invoked_subcommand is not None:
        return

    if not notebook:
        typer.echo("Error: --notebook is required for full smelt run.")
        raise typer.Exit(1)

    try:
        result = assay_module.infer_semantic_types(str(notebook), df_name=df)
    except ValueError as e:
        typer.echo(f"Error: {e}")
        typer.echo("Hint: Run 'nb-smith assay profile' first, execute cells, then retry.")
        raise typer.Exit(1)

    smelt_module.apply_fixes(str(notebook), result["findings"], result["normalization"])


@smelt_app.command()
def dates(
    notebook: Path = typer.Argument(..., help="Target notebook (.ipynb)"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
    cols: Optional[str] = typer.Option(None, "--cols", help="Comma-separated column names"),
) -> None:
    """Insert pd.to_datetime() coercion cells."""
    smelt_module.insert_date_fixes(str(notebook), df_name=df, columns=_parse_cols(cols))


@smelt_app.command()
def numerics(
    notebook: Path = typer.Argument(..., help="Target notebook (.ipynb)"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
    cols: Optional[str] = typer.Option(None, "--cols", help="Comma-separated column names"),
) -> None:
    """Insert pd.to_numeric() coercion cells."""
    smelt_module.insert_numeric_fixes(str(notebook), df_name=df, columns=_parse_cols(cols))


@smelt_app.command()
def booleans(
    notebook: Path = typer.Argument(..., help="Target notebook (.ipynb)"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
    cols: Optional[str] = typer.Option(None, "--cols", help="Comma-separated column names"),
) -> None:
    """Insert boolean mapping cells (yes/no/true/false/1/0)."""
    smelt_module.insert_boolean_fixes(str(notebook), df_name=df, columns=_parse_cols(cols))


@smelt_app.command()
def categoricals(
    notebook: Path = typer.Argument(..., help="Target notebook (.ipynb)"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
    cols: Optional[str] = typer.Option(None, "--cols", help="Comma-separated column names"),
) -> None:
    """Insert astype('category') coercion cells."""
    smelt_module.insert_categorical_fixes(str(notebook), df_name=df, columns=_parse_cols(cols))


@smelt_app.command()
def normalize(
    notebook: Path = typer.Argument(..., help="Target notebook (.ipynb)"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
) -> None:
    """Insert df.rename() cells for snake_case column normalization."""
    smelt_module.insert_normalization(str(notebook), df_name=df)


@smelt_app.command()
def trim(
    notebook: Path = typer.Argument(..., help="Target notebook (.ipynb)"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
    cols: Optional[str] = typer.Option(None, "--cols", help="Comma-separated column names"),
) -> None:
    """Insert .str.strip() cells for whitespace removal."""
    smelt_module.insert_trim_fixes(str(notebook), df_name=df, columns=_parse_cols(cols))


app.add_typer(smelt_app, name="smelt")

# ═══════════════════════════════════════════════════════════════════════════════
# CUPEL
# ═══════════════════════════════════════════════════════════════════════════════


@cupel_app.callback(invoke_without_command=True)
def cupel_full(
    ctx: typer.Context,
    notebook: Optional[Path] = typer.Option(None, "--notebook", "-n", help="Target notebook (.ipynb)"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
) -> None:
    """Phase 3: Deep diagnostics. Insert helper cells and run missingness,
    outlier, duplicate, cardinality, and skew analysis."""
    if ctx.invoked_subcommand is not None:
        return

    if not notebook:
        typer.echo("Error: --notebook is required for full cupel run.")
        raise typer.Exit(1)

    cupel_module.analyze_missingness(str(notebook), df_name=df)
    cupel_module.analyze_outliers(str(notebook), df_name=df)
    cupel_module.analyze_duplicates(str(notebook), df_name=df)
    cupel_module.analyze_cardinality(str(notebook), df_name=df)
    cupel_module.analyze_skew(str(notebook), df_name=df)
    typer.echo("[cupel] All diagnostics inserted.")


@cupel_app.command()
def missingness(
    notebook: Path = typer.Argument(..., help="Target notebook (.ipynb)"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
) -> None:
    """Insert missingness analysis cells. MCAR test, MAR detection, MNAR
    candidate flagging."""
    cupel_module.analyze_missingness(str(notebook), df_name=df)


@cupel_app.command()
def outliers(
    notebook: Path = typer.Argument(..., help="Target notebook (.ipynb)"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
) -> None:
    """Insert outlier analysis cells. IQR (1.5x / 3.0x) and Z-score (>3)
    detection."""
    cupel_module.analyze_outliers(str(notebook), df_name=df)


@cupel_app.command()
def duplicates(
    notebook: Path = typer.Argument(..., help="Target notebook (.ipynb)"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
) -> None:
    """Insert duplicate analysis cells. Duplicate row detection with
    statistical association testing."""
    cupel_module.analyze_duplicates(str(notebook), df_name=df)


@cupel_app.command()
def cardinality(
    notebook: Path = typer.Argument(..., help="Target notebook (.ipynb)"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
) -> None:
    """Insert cardinality analysis cells. Distinct value counts and
    percentage per column."""
    cupel_module.analyze_cardinality(str(notebook), df_name=df)


@cupel_app.command()
def skew(
    notebook: Path = typer.Argument(..., help="Target notebook (.ipynb)"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
) -> None:
    """Insert skewness analysis cells. Skewness analysis for numeric
    columns."""
    cupel_module.analyze_skew(str(notebook), df_name=df)


app.add_typer(cupel_app, name="cupel")

# ═══════════════════════════════════════════════════════════════════════════════
# CAST / FORGE / EXECUTE
# ═══════════════════════════════════════════════════════════════════════════════


@app.command(rich_help_panel=ANALYSIS_PANEL, hidden=True)
def cast(
    notebook: Path = typer.Argument(..., help="Target notebook (.ipynb)"),
) -> None:
    """Pipeline synthesis (not yet implemented)."""
    try:
        cast_build_pipeline(str(notebook))
    except NotImplementedError:
        typer.echo("Pipeline synthesis is not yet implemented.")


@app.command(rich_help_panel=ALL_IN_ONE_PANEL)
def forge(
    from_dir: Path = typer.Option(..., "--from", help="Data directory to scan"),
    output: Path = typer.Option(..., "--output", "-o", help="Output notebook path"),
    df: Optional[str] = typer.Option(None, "--df", help="Target specific DataFrame"),
) -> None:
    """Run all phases: scan, profile, execute, assay, smelt, cupel, execute.
    One command from raw data to fully diagnosed notebook."""
    typer.echo(f"[forge] 1/7 Scanning {from_dir} ...")
    assay_module.scan(str(from_dir), str(output))

    typer.echo("[forge] 2/7 Profiling DataFrames ...")
    assay_module.profile_dataframe(str(output), df_name=df)

    typer.echo("[forge] 3/7 Executing notebook ...")
    execute_notebook(str(output))

    typer.echo("[forge] 4/7 Running semantic type inference ...")
    try:
        result = assay_module.infer_semantic_types(str(output), df_name=df)
    except ValueError as e:
        typer.echo(f"Error during inference: {e}")
        raise typer.Exit(1)

    typer.echo("[forge] 5/7 Applying type fixes ...")
    smelt_module.apply_fixes(str(output), result["findings"], result["normalization"])

    typer.echo("[forge] 6/7 Inserting diagnostics ...")
    cupel_module.analyze_missingness(str(output), df_name=df)
    cupel_module.analyze_outliers(str(output), df_name=df)
    cupel_module.analyze_duplicates(str(output), df_name=df)
    cupel_module.analyze_cardinality(str(output), df_name=df)
    cupel_module.analyze_skew(str(output), df_name=df)

    typer.echo("[forge] 7/7 Executing final notebook ...")
    execute_notebook(str(output))

    typer.echo(f"[forge] Done: {output}")


@app.command(rich_help_panel=UTILITY_PANEL)
def execute(
    notebook: Path = typer.Argument(..., help="Target notebook (.ipynb)"),
    timeout: int = typer.Option(300, "--timeout", help="Execution timeout in seconds"),
) -> None:
    """Execute all cells in a notebook via nb-cli. Required between phases
    that parse prior cell outputs (e.g., profile must execute before infer
    can read outputs)."""
    execute_notebook(str(notebook), timeout=timeout)
    typer.echo(f"Executed: {notebook}")
