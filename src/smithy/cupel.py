"""
Deep Diagnostics (Auditing)

- Missingness: %NaNs, MCAR/MAR hints, pattern flags
- Outliers: IQR/Z-score rules
- Duplicates, high-cardinality, skew
- Output: standardized severity sentences + diagnostic tags
"""

from smithy._shared import (
    dedent_block,
    insert_analysis_cells,
    resolve_df_names,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Missingness
# ═══════════════════════════════════════════════════════════════════════════════


def _generate_missingness_code() -> str:
    """Generate missingness analysis helper code for notebook cells."""
    return dedent_block("""
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

        def _missingness_print_section(title):
            line = "=" * 70
            print(f"\\n{line}\\n{title}\\n{line}")

        def _missingness_summarize(df):
            missing_counts = df.isnull().sum()
            missing_percent = (missing_counts / len(df)) * 100
            summary = pd.DataFrame({
                "Missing Count": missing_counts,
                "Missing %": missing_percent.round(2),
                "Entirely Missing": missing_counts == len(df),
            }).sort_values("Missing %", ascending=False)
            _missingness_print_section("Missingness Summary")
            print(summary)
            return summary

        def _missingness_fully_missing(df):
            return set(df.columns[df.isnull().all()])

        def _missingness_mcar(df):
            numeric_df = df.select_dtypes(include=[np.number])
            if not HAS_MCAR:
                return None, "pyampute not installed"
            if numeric_df.empty or numeric_df.shape[1] < 2:
                return None, "need at least 2 numeric columns"
            if numeric_df.isnull().sum().sum() == 0:
                return None, "no missing values in numeric columns"

            note = " (numeric only — categoricals excluded)" if (
                df.shape[1] > numeric_df.shape[1]
            ) else ""

            try:
                p = MCARTest().little_mcar_test(numeric_df)
                _missingness_print_section(f"Little's MCAR Test{note}")
                print(f"p-value: {p:.6f}")
                return p, "ok"
            except Exception as e:
                return None, f"test failed: {e}"

        def _missingness_collect_mar(df, missing_mask, cols_with_na, skip_cols):
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

        def _missingness_detect_mar(df, cols_with_na, skip_cols):
            _missingness_print_section("MAR Analysis (with multiple-testing correction)")
            missing_mask = df.isnull().astype(int)
            records = _missingness_collect_mar(df, missing_mask, cols_with_na, skip_cols)

            if not records:
                print("No testable pairs found (insufficient group sizes).")
                return set()

            p_raw = [r["p_raw"] for r in records]
            reject, p_corrected, _, _ = multipletests(p_raw, method="fdr_bh", alpha=ALPHA)

            mar_detected = set()
            for r, rej, pc in zip(records, reject, p_corrected):
                if rej:
                    mar_detected.add(r["target"])
                    print(f"MAR: '{r['target']}' ~ '{r['feature']}' "
                          f"({r['test']}, p_corrected={pc:.4f})")

            if not mar_detected:
                print("No MAR relationships detected after correction.")
            return mar_detected

        def analyze_missingness(df, name="DataFrame"):
            _missingness_print_section(f"Missingness Analysis: {name}")
            _missingness_summarize(df)

            skip = _missingness_fully_missing(df)
            if skip:
                print(f"\\nEntirely missing — excluded from MAR/MNAR: {skip}")

            cols_with_na = df.columns[df.isnull().any() & ~df.columns.isin(skip)]
            if cols_with_na.empty:
                print("\\nNo partial missingness found.")
                return

            mcar_p, note = _missingness_mcar(df)

            if mcar_p is not None and mcar_p >= ALPHA:
                print(f"\\nMCAR: p = {mcar_p:.4f}")
                return

            if mcar_p is None:
                print(f"\\nMCAR unavailable: {note}")
            else:
                print(f"\\nMCAR: p = {mcar_p:.4f}")

            mar_detected = _missingness_detect_mar(df, cols_with_na, skip_cols=skip)
            mnar_candidates = set(cols_with_na) - mar_detected

            if mnar_candidates:
                _missingness_print_section("MNAR Candidates")
                for col in mnar_candidates:
                    print(f"MNAR: '{col}' — no significant associations detected.")
    """)


def analyze_missingness(notebook_path: str, df_name: str | None = None) -> None:
    """Add missingness analysis cells to a notebook."""
    df_names = resolve_df_names(notebook_path, df_name)
    insert_analysis_cells(notebook_path, df_names, _generate_missingness_code, "missingness")


# ═══════════════════════════════════════════════════════════════════════════════
# Outliers
# ═══════════════════════════════════════════════════════════════════════════════


def _generate_outlier_code() -> str:
    """Generate outlier analysis helper code for notebook cells."""
    return dedent_block("""
        import numpy as np
        import pandas as pd

        ALPHA_IQR_MILD = 1.5
        ALPHA_IQR_EXTREME = 3.0
        ZSCORE_THRESHOLD = 3.0

        def _outlier_print_section(title):
            line = "=" * 70
            print(f"\\n{line}\\n{title}\\n{line}")

        def _outlier_summary(iqr_result):
            mild = iqr_result["mild_count"]
            extreme = iqr_result["extreme_count"]
            if mild == 0 and extreme == 0:
                return "0 outliers"
            upper = iqr_result["mild_upper"]
            lower = iqr_result["mild_lower"]
            if upper > lower:
                tail = "upper tail"
            elif lower > upper:
                tail = "lower tail"
            else:
                tail = "both tails"
            return f"{mild} mild, {extreme} extreme ({tail})"

        def _outlier_iqr(series, col_name):
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

        def _outlier_zscore(series, col_name):
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
            _outlier_print_section(f"Outlier Analysis: {name}")

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            other_cols = [c for c in df.columns if c not in numeric_cols]

            for col in numeric_cols:
                iqr = _outlier_iqr(df[col], col)
                z = _outlier_zscore(df[col], col)
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

                print(f"Summary: {_outlier_summary(iqr)}")

            for col in other_cols:
                print(f"\n--- Column: {col} ({df[col].dtype}) ---")
                print("Skipped: non-numeric column")
    """)


def analyze_outliers(notebook_path: str, df_name: str | None = None) -> None:
    """Add outlier analysis cells to a notebook."""
    df_names = resolve_df_names(notebook_path, df_name)
    insert_analysis_cells(
        notebook_path,
        df_names,
        _generate_outlier_code,
        "outlier",
        call_template="analyze_outliers({df}, name='{df}')",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Duplicates
# ═══════════════════════════════════════════════════════════════════════════════


def _generate_duplicate_code() -> str:
    """Generate duplicate analysis helper code for notebook cells."""
    return dedent_block("""
        import numpy as np
        import pandas as pd
        from scipy.stats import mannwhitneyu, chi2_contingency

        ALPHA = 0.05

        def _duplicate_print_section(title):
            line = "=" * 70
            print(f"\\n{line}\\n{title}\\n{line}")

        def _duplicate_summary(dup_count, dup_pct):
            if dup_count == 0:
                return "0 duplicates"
            return f"{dup_count} duplicates ({dup_pct:.2f}%)"

        def analyze_duplicates(df, name="DataFrame"):
            _duplicate_print_section(f"Duplicate Analysis: {name}")

            total = len(df)
            dup_mask_extra = df.duplicated(keep="first")
            dup_count = int(dup_mask_extra.sum())
            dup_pct = (dup_count / total * 100) if total else 0

            print(f"\\nTotal rows: {total}")
            print(f"Duplicate rows (keep='first'): {dup_count} ({dup_pct:.2f}%)")

            if dup_count > 0:
                is_dup = df.duplicated(keep=False).astype(int)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                categorical_cols = df.select_dtypes(include=["object", "category"]).columns
                sig_assocs = []

                for col in numeric_cols:
                    grp_dup = df.loc[is_dup == 1, col].dropna()
                    grp_unique = df.loc[is_dup == 0, col].dropna()
                    if len(grp_dup) < 5 or len(grp_unique) < 5:
                        continue
                    try:
                        _, p = mannwhitneyu(grp_dup, grp_unique, alternative="two-sided")
                        if p < ALPHA:
                            mean_dup = grp_dup.mean()
                            mean_unique = grp_unique.mean()
                            sig_assocs.append({
                                "col": col, "type": "numeric",
                                "test": "mannwhitney", "p": p,
                                "note": (f"distribution difference in '{col}' "
                                         f"(dup mean={mean_dup:.2f}, unique mean={mean_unique:.2f})")
                            })
                    except Exception:
                        pass

                for col in categorical_cols:
                    try:
                        contingency = pd.crosstab(is_dup, df[col])
                        if contingency.shape[1] < 2:
                            continue
                        _, p, _, _ = chi2_contingency(contingency)
                        if p < ALPHA:
                            rates = {}
                            for cat in df[col].dropna().unique():
                                mask_cat = df[col] == cat
                                n_cat = mask_cat.sum()
                                n_dup_cat = (is_dup[mask_cat] == 1).sum()
                                rates[cat] = (n_dup_cat / n_cat * 100) if n_cat else 0
                            top_rate = max(rates.items(), key=lambda x: x[1])
                            sig_assocs.append({
                                "col": col, "type": "categorical",
                                "test": "chi2", "p": p,
                                "note": (f"association with '{col}' "
                                         f"(highest rate: '{top_rate[0]}'={top_rate[1]:.1f}% dup)")
                            })
                    except Exception:
                        pass

                if sig_assocs:
                    print(f"\nAssociations with duplicates:")
                    for assoc in sorted(sig_assocs, key=lambda x: x['p']):
                        print(f"  {assoc['type']}: {assoc['note']} (p={assoc['p']:.4f})")

            print(f"\n{_duplicate_summary(dup_count, dup_pct)}")
    """)


def analyze_duplicates(notebook_path: str, df_name: str | None = None) -> None:
    """Add duplicate analysis cells to a notebook."""
    df_names = resolve_df_names(notebook_path, df_name)
    insert_analysis_cells(notebook_path, df_names, _generate_duplicate_code, "duplicates")


# ═══════════════════════════════════════════════════════════════════════════════
# Cardinality
# ═══════════════════════════════════════════════════════════════════════════════


def _generate_cardinality_code() -> str:
    """Generate cardinality analysis helper code for notebook cells."""
    return dedent_block("""
        import pandas as pd

        def _cardinality_print_section(title):
            line = "=" * 70
            print(f"\\n{line}\\n{title}\\n{line}")

        def analyze_cardinality(df, name="DataFrame"):
            _cardinality_print_section(f"Cardinality Analysis: {name}")

            total = len(df)
            print(f"\nTotal rows: {total}")
            print(f"{'Column':<20} {'Distinct':>8} {'Distinct %':>10}")
            print("-" * 60)

            records = []
            for col in df.columns:
                n_unique = df[col].nunique(dropna=False)
                pct_unique = (n_unique / total * 100) if total else 0
                records.append({
                    "Column": col,
                    "Distinct": n_unique,
                    "Distinct %": round(pct_unique, 2),
                })

            for r in sorted(records, key=lambda x: x["Distinct %"], reverse=True):
                print(f"{r['Column']:<20} {r['Distinct']:>8} {r['Distinct %']:>10.2f}")
    """)


def analyze_cardinality(notebook_path: str, df_name: str | None = None) -> None:
    """Add cardinality analysis cells to a notebook."""
    df_names = resolve_df_names(notebook_path, df_name)
    insert_analysis_cells(notebook_path, df_names, _generate_cardinality_code, "cardinality")


# ═══════════════════════════════════════════════════════════════════════════════
# Skew
# ═══════════════════════════════════════════════════════════════════════════════


def _generate_skew_code() -> str:
    """Generate skewness analysis helper code for notebook cells."""
    return dedent_block("""
        import numpy as np
        import pandas as pd
        from scipy.stats import skew

        def _skew_print_section(title):
            line = "=" * 70
            print(f"\\n{line}\\n{title}\\n{line}")

        def _skew_label(skew_val):
            direction = "right" if skew_val > 0 else "left"
            if abs(skew_val) < 0.5:
                return f"skew = {skew_val:.2f}"
            return f"skew = {skew_val:.2f} ({direction})"

        def analyze_skew(df, name="DataFrame"):
            _skew_print_section(f"Skew Analysis: {name}")

            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                print("No numeric columns found.")
                return

            print(f"{'Column':<20} {'Skew':>10}")
            print("-" * 40)

            for col in numeric_cols:
                s = df[col].dropna()
                if s.empty or s.nunique() <= 1:
                    print(f"{col:<20} {'—':>10}  constant or empty — skipped")
                    continue

                sk = skew(s, bias=False)
                print(f"{col:<20} {sk:>10.2f}  {_skew_label(sk)}")
    """)


def analyze_skew(notebook_path: str, df_name: str | None = None) -> None:
    """Add skewness analysis cells to a notebook."""
    df_names = resolve_df_names(notebook_path, df_name)
    insert_analysis_cells(notebook_path, df_names, _generate_skew_code, "skew")
