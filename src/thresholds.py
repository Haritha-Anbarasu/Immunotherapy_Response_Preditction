"""
Phase 7b: Cross-cancer-type threshold comparison -- the core novel-
contribution output of this project.

Takes the per-cancer-type SHAP thresholds produced by
`shap_analysis.py --per-cancer-type` and asks, for each feature that has a
threshold in at least two cancer types: how much does the threshold vary?

A feature whose threshold is nearly identical across cancer types is a
candidate "universal" biomarker cutoff. A feature whose threshold swings
widely is cancer-type-specific -- clinically important to know before anyone
tries to apply a single cutoff pan-cancer.
"""
import argparse
import os
import numpy as np
import pandas as pd


def compare_thresholds_across_cancer_types(thresholds_df: pd.DataFrame) -> pd.DataFrame:
    """
    `thresholds_df` expected columns: feature, threshold, cancer_type
    (i.e. the output of shap_analysis.py --per-cancer-type, concatenated).
    """
    pivot = thresholds_df.dropna(subset=["threshold"]).pivot_table(
        index="feature", columns="cancer_type", values="threshold"
    )

    summary = pd.DataFrame({
        "n_cancer_types_with_threshold": pivot.notna().sum(axis=1),
        "mean_threshold": pivot.mean(axis=1),
        "std_threshold": pivot.std(axis=1),
        "coefficient_of_variation": pivot.std(axis=1) / pivot.mean(axis=1).abs(),
        "min_threshold": pivot.min(axis=1),
        "max_threshold": pivot.max(axis=1),
    })

    summary = summary[summary["n_cancer_types_with_threshold"] >= 2]
    summary["classification"] = np.where(
        summary["coefficient_of_variation"] < 0.25, "likely_universal",
        np.where(summary["coefficient_of_variation"] < 0.75, "moderately_variable",
                 "cancer_type_specific")
    )

    return pivot.join(summary).sort_values("coefficient_of_variation")


def main():
    parser = argparse.ArgumentParser(description="Compare SHAP thresholds across cancer types.")
    parser.add_argument("--shap-dir", required=True, help="Directory from shap_analysis.py --per-cancer-type")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    path = os.path.join(args.shap_dir, "thresholds_by_cancer_type.csv")
    thresholds_df = pd.read_csv(path)

    comparison = compare_thresholds_across_cancer_types(thresholds_df)
    comparison.to_csv(args.output)

    n_universal = (comparison["classification"] == "likely_universal").sum()
    n_specific = (comparison["classification"] == "cancer_type_specific").sum()
    print(f"[ok] wrote {args.output}")
    print(f"[info] {n_universal} feature(s) look pan-cancer universal, "
          f"{n_specific} feature(s) look cancer-type-specific -- "
          f"this table is your project's headline result.")


if __name__ == "__main__":
    main()
