"""
Deep Diagnostics (Auditing)

- Missingness: %NaNs, MCAR/MAR hints (null corr w/other cols), pattern flags
- Outliers: IQR/Z-score rules
- Duplicates, high-cardinality, skew, target leakage (corr>0.9)
- Output: issue table (type|col|severity|fix_snippet) + asserts
"""

import textwrap
from tools.notebook import (create_notebook,
                            insert_cell,
                            read_notebook,
                            insert_cells_batch,
                            find_dataframe_cells)

def _generate_profile_code() -> str:
    """Generate profiling code for a DataFrame including MCAR/MAR/MNAR logic."""
    return textwrap.dedent("""
        import numpy as np
        import pandas as pd
        from scipy.stats import mannwhitneyu, chi2_contingency
        from statsmodels.stats.multitest import multipletests

        try:
            from pyampute.exploration.mcar_statistical_tests import MCARTest
            HAS_MCAR = True
        except ImportError:
            HAS_MCAR = False

        MIN_GROUP_SIZE = 30
        ALPHA = 0.05

        def print_section(title):
            line = "=" * 70
            print(f"\\n{line}\\n{title}\\n{line}")

        def summarize_missingness(df):
            missing_counts = df.isnull().sum()
            missing_percent = (missing_counts / len(df)) * 100
            summary = pd.DataFrame({
                "Missing Count": missing_counts,
                "Missing %": missing_percent.round(2),
                "Entirely Missing": missing_counts == len(df),
            }).sort_values("Missing %", ascending=False)
            print_section("Missingness Summary")
            print(summary)
            return summary

        def fully_missing_cols(df):
            return set(df.columns[df.isnull().all()])

        def run_mcar_test(df):
            numeric_df = df.select_dtypes(include=[np.number])
            if not HAS_MCAR:
                return None, "pyampute not installed"
            if numeric_df.empty or numeric_df.shape[1] < 2:
                return None, "need at least 2 numeric columns"
            if numeric_df.isnull().sum().sum() == 0:
                return None, "no missing values in numeric columns"

            warn = " (numeric only — categoricals excluded)" if (
                df.shape[1] > numeric_df.shape[1]
            ) else ""

            try:
                p = MCARTest().little_mcar_test(numeric_df)
                print_section(f"Little's MCAR Test{warn}")
                print(f"p-value: {p:.6f}")
                return p, "ok"
            except Exception as e:
                return None, f"test failed: {e}"

        def collect_mar_pvalues(df, missing_mask, cols_with_na, skip_cols):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns
            records = []

            for target_col in cols_with_na:
                if target_col in skip_cols:
                    continue
                mask = missing_mask[target_col]

                for feature in numeric_cols:
                    if feature == target_col:
                        continue
                    grp_miss = df.loc[mask == 1, feature].dropna()
                    grp_obs  = df.loc[mask == 0, feature].dropna()
                    if len(grp_miss) < MIN_GROUP_SIZE or len(grp_obs) < MIN_GROUP_SIZE:
                        continue
                    try:
                        _, p = mannwhitneyu(grp_miss, grp_obs, alternative="two-sided")
                        records.append({"target": target_col, "feature": feature,
                                        "test": "mannwhitney", "p_raw": p})
                    except Exception:
                        pass

                for feature in categorical_cols:
                    if feature == target_col:
                        continue
                    try:
                        contingency = pd.crosstab(mask, df[feature])
                        if contingency.shape[1] < 2:
                            continue
                        _, p, _, _ = chi2_contingency(contingency)
                        records.append({"target": target_col, "feature": feature,
                                        "test": "chi2", "p_raw": p})
                    except Exception:
                        pass

            return records

        def detect_mar(df, cols_with_na, skip_cols):
            print_section("MAR Analysis (with multiple-testing correction)")
            missing_mask = df.isnull().astype(int)
            records = collect_mar_pvalues(df, missing_mask, cols_with_na, skip_cols)

            if not records:
                print("No testable pairs found — check group sizes.")
                return set()

            p_raw = [r["p_raw"] for r in records]
            reject, p_corrected, _, _ = multipletests(p_raw, method="fdr_bh", alpha=ALPHA)

            mar_detected = set()
            for r, rej, pc in zip(records, reject, p_corrected):
                if rej:
                    mar_detected.add(r["target"])
                    print(f"[MAR] '{r['target']}' ~ '{r['feature']}' "
                          f"({r['test']}, p_corrected={pc:.4f})")

            if not mar_detected:
                print("No MAR relationships detected after correction.")
            return mar_detected

        def analyze_missingness(df, name="DataFrame"):
            print_section(f"Missingness Analysis: {name}")
            summarize_missingness(df)

            skip = fully_missing_cols(df)
            if skip:
                print(f"\\n[Warning] Entirely missing — excluded from MAR/MNAR: {skip}")

            cols_with_na = df.columns[df.isnull().any() & ~df.columns.isin(skip)]
            if cols_with_na.empty:
                print("\\n[Conclusion] No partial missingness found.")
                return

            mcar_p, note = run_mcar_test(df)

            if mcar_p is not None and mcar_p >= ALPHA:
                print(f"\\n[Conclusion] Data may be MCAR (p={mcar_p:.4f}).")
                return

            if mcar_p is None:
                print(f"\\n[MCAR] Unavailable: {note}. Proceeding with MAR analysis.")
            else:
                print(f"\\n[Conclusion] Likely NOT MCAR (p={mcar_p:.4f}). Running MAR analysis.")

            mar_detected = detect_mar(df, cols_with_na, skip_cols=skip)
            mnar_candidates = set(cols_with_na) - mar_detected

            if mnar_candidates:
                print_section("MNAR Candidates")
                for col in mnar_candidates:
                    print(f"[MNAR?] '{col}' — no MCAR/MAR pattern. Inspect domain context.")
    """).strip()


def analyze_missingness(notebook_path: str, df_name: str = None):
    nb = read_notebook(notebook_path)

    if df_name:
        df_names = [df_name]
    else:
        df_cells = find_dataframe_cells(nb)
        df_names = [name for _, name in df_cells if name and name not in ('null_df', 'unique_counts')]

    if not df_names:
        raise ValueError("No DataFrames found in notebook")

    insert_cell(notebook_path, _generate_profile_code(), cell_type='code')

    cells_to_insert = [
        {"code": f"analyze_missingness({df}, name='{df}')", "type": "code"}
        for df in df_names
    ]

    insert_cells_batch(notebook_path, cells_to_insert)
    print(f"Added {len(cells_to_insert)} profile cells")

def _generate_outlier_code() -> str:
    """Generate profiling code for outlier detection using IQR and Z-score."""
    return textwrap.dedent("""
        import numpy as np
        import pandas as pd

        ALPHA_IQR_MILD = 1.5
        ALPHA_IQR_EXTREME = 3.0
        ZSCORE_THRESHOLD = 3.0

        def print_section(title):
            line = "=" * 70
            print(f"\\n{line}\\n{title}\\n{line}")

        def _severity_sentence(iqr_result, zscore_result):
            \"\"\"Generate a one-sentence summary from outlier counts.\"\"\""
            mild = iqr_result["mild_count"]
            extreme = iqr_result["extreme_count"]
            total_rows = iqr_result["total_rows"]
            pct_mild = (mild / total_rows * 100) if total_rows else 0
            pct_extreme = (extreme / total_rows * 100) if total_rows else 0

            if mild == 0 and extreme == 0:
                return "low — no outliers detected, distribution appears normal"

            if pct_extreme > 1 or pct_mild > 5:
                tail = ""
                if iqr_result["mild_upper"] > iqr_result["mild_lower"]:
                    tail = "heavy-tailed with extreme upper values"
                elif iqr_result["mild_lower"] > iqr_result["mild_upper"]:
                    tail = "heavy-tailed with extreme lower values"
                else:
                    tail = "heavy-tailed on both ends"
                return f"high — {pct_mild:.1f}% of rows flagged, possible data quality issue or {tail}"

            upper = iqr_result["mild_upper"]
            lower = iqr_result["mild_lower"]
            if upper > lower:
                return "moderate — right-skewed with long upper tail, inspect extreme values"
            elif lower > upper:
                return "moderate — left-skewed with long lower tail, inspect extreme values"
            else:
                return "moderate — outliers present on both tails, inspect distribution"

        def iqr_outliers(series, col_name):
            \"\"\"Detect mild and extreme outliers using IQR.\"\"\""
            s = series.dropna()
            total_rows = len(series)

            if s.empty or s.nunique() <= 1:
                return {
                    "mild_count": 0, "extreme_count": 0,
                    "mild_lower": 0, "mild_upper": 0,
                    "extreme_lower": 0, "extreme_upper": 0,
                    "total_rows": total_rows,
                }

            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1

            mild_low = q1 - ALPHA_IQR_MILD * iqr
            mild_high = q3 + ALPHA_IQR_MILD * iqr
            extreme_low = q1 - ALPHA_IQR_EXTREME * iqr
            extreme_high = q3 + ALPHA_IQR_EXTREME * iqr

            mild_mask = (s < mild_low) | (s > mild_high)
            extreme_mask = (s < extreme_low) | (s > extreme_high)

            return {
                "mild_count": int(mild_mask.sum()),
                "extreme_count": int(extreme_mask.sum()),
                "mild_lower": int((s < mild_low).sum()),
                "mild_upper": int((s > mild_high).sum()),
                "extreme_lower": int((s < extreme_low).sum()),
                "extreme_upper": int((s > extreme_high).sum()),
                "total_rows": total_rows,
                "bounds": {
                    "mild": (mild_low, mild_high),
                    "extreme": (extreme_low, extreme_high),
                }
            }

        def zscore_outliers(series, col_name):
            \"\"\"Detect outliers using classic mean/std Z-score.\"\"\""
            s = series.dropna()
            total_rows = len(series)

            if s.empty or s.std() == 0:
                return {"outlier_count": 0, "max_z": 0.0, "total_rows": total_rows}

            z = np.abs((s - s.mean()) / s.std())
            outlier_mask = z > ZSCORE_THRESHOLD

            return {
                "outlier_count": int(outlier_mask.sum()),
                "max_z": float(z.max()),
                "total_rows": total_rows,
            }

        def analyze_outliers(df, name="DataFrame"):
            print_section(f"Outlier Analysis: {name}")

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            other_cols = [c for c in df.columns if c not in numeric_cols]

            for col in numeric_cols:
                iqr = iqr_outliers(df[col], col)
                z = zscore_outliers(df[col], col)
                total = iqr["total_rows"]

                print(f"\\n--- Column: {col} (numeric) ---")

                if iqr["mild_count"] == 0:
                    print("IQR: no outliers")
                else:
                    pct_mild = iqr["mild_count"] / total * 100
                    pct_extreme = iqr["extreme_count"] / total * 100
                    print(f"IQR: {iqr['mild_count']} mild ({pct_mild:.1f}%), {iqr['extreme_count']} extreme ({pct_extreme:.1f}%)")
                    b = iqr.get("bounds", {})
                    if "mild" in b:
                        print(f"  Bounds: mild [{b['mild'][0]:.2f}, {b['mild'][1]:.2f}] | extreme [{b['extreme'][0]:.2f}, {b['extreme'][1]:.2f}]")

                if z["outlier_count"] == 0:
                    print("Z-score: no outliers")
                else:
                    pct_z = z["outlier_count"] / total * 100
                    print(f"Z-score: {z['outlier_count']} outliers ({pct_z:.1f}%), max |z| = {z['max_z']:.2f}")

                severity = _severity_sentence(iqr, z)
                print(f"Severity: {severity}")

            for col in other_cols:
                print(f"\\n--- Column: {col} ({df[col].dtype}) ---")
                print("Skipped: non-numeric column")
    """).strip()


def analyze_outliers(notebook_path: str, df_name: str = None):
    """Add outlier analysis cells to a notebook."""
    nb = read_notebook(notebook_path)

    if df_name:
        df_names = [df_name]
    else:
        df_cells = find_dataframe_cells(nb)
        df_names = [name for _, name in df_cells if name and name not in ('null_df', 'unique_counts')]

    if not df_names:
        raise ValueError("No DataFrames found in notebook")

    insert_cell(notebook_path, _generate_outlier_code(), cell_type='code')

    cells_to_insert = [
        {"code": f"analyze_outliers({df}, name='{df}')", "type": "code"}
        for df in df_names
    ]

    insert_cells_batch(notebook_path, cells_to_insert)
    print(f"Added {len(cells_to_insert)} outlier analysis cells")


def analyze_duplicates():
    return


def analyze_cardinality():
    return


def analyze_skew():
    return


def generate_report():
    return

