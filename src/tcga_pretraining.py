"""
Phase 3 (optional/stretch): EaSIeR-style label-free pretraining.

Idea: ICI-response-labeled cohorts are small (tens-hundreds of patients).
TCGA has thousands of patients with RNA-seq but (mostly) no ICI response
labels. Instead of training directly on the small labeled cohorts, first
train a model on TCGA to predict a general "immune activity" score derived
from established immune signatures (no outcome label needed), then use that
model's predicted score as an ADDITIONAL ENGINEERED FEATURE when training
the final response classifier on the small labeled cohorts.

This lets the small labeled cohorts benefit from patterns learned across
thousands of TCGA samples, without needing TCGA patients to have known ICI
outcomes (most don't -- TCGA predates widespread ICI use for most patients).

Reference immune-activity target: the same IFN-gamma / cytolytic-score style
signatures used in feature_engineering.py, but computed at TCGA pan-cancer
scale and used as a *regression target* to pretrain a feature extractor.
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from src.feature_engineering import compute_cytolytic_score, compute_ifng_signature


def build_tcga_immune_activity_target(tcga_expression: pd.DataFrame) -> pd.Series:
    """
    Construct a label-free 'immune activity' target by averaging the
    (standardized) cytolytic score and IFN-gamma signature -- both are
    well-established measures of anti-tumor immune activity that don't
    require any treatment-outcome data to compute.
    """
    cytolytic = compute_cytolytic_score(tcga_expression)
    ifng = compute_ifng_signature(tcga_expression)

    z_cytolytic = (cytolytic - cytolytic.mean()) / cytolytic.std()
    z_ifng = (ifng - ifng.mean()) / ifng.std()
    return (z_cytolytic + z_ifng) / 2


def pretrain_on_tcga(tcga_expression: pd.DataFrame, gene_cols: list,
                      random_state: int = 42):
    """
    Train a RandomForestRegressor on TCGA to predict the label-free immune
    activity target from raw gene expression. Returns the fitted model,
    which can then be applied to your (much smaller) ICI-labeled cohorts to
    generate a "TCGA-informed immune activity score" feature.
    """
    target = build_tcga_immune_activity_target(tcga_expression)
    X = tcga_expression[gene_cols].fillna(tcga_expression[gene_cols].median())
    valid = target.notna()
    X, target = X[valid], target[valid]

    model = RandomForestRegressor(n_estimators=300, random_state=random_state, n_jobs=-1)
    scores = cross_val_score(model, X, target, cv=5, scoring="r2")
    print(f"[info] TCGA pretraining 5-fold CV R^2: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

    model.fit(X, target)
    return model


def apply_pretrained_score(model, target_expression: pd.DataFrame, gene_cols: list) -> pd.Series:
    """Apply the TCGA-pretrained model to a (small) labeled ICI cohort."""
    available = [g for g in gene_cols if g in target_expression.columns]
    missing = set(gene_cols) - set(available)
    X = target_expression.reindex(columns=gene_cols, fill_value=np.nan)
    X = X.fillna(X.median())
    if missing:
        print(f"[warn] {len(missing)} genes used in TCGA pretraining are missing "
              f"in the target cohort and were imputed with the cohort median.")
    return pd.Series(model.predict(X), index=target_expression.index, name="tcga_immune_activity_score")


def main():
    parser = argparse.ArgumentParser(description="EaSIeR-style TCGA label-free pretraining.")
    parser.add_argument("--tcga-dir", required=True, help="Directory containing a TCGA expression CSV")
    parser.add_argument("--target-cohort", default=None,
                         help="Optional: apply the pretrained model to this cohort's expression CSV")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import glob
    tcga_files = glob.glob(f"{args.tcga_dir}/*.csv")
    if not tcga_files:
        raise FileNotFoundError(f"No CSV files found in {args.tcga_dir}. "
                                 "Download TCGA expression matrices via the GDC Data Portal first.")

    tcga_expr = pd.concat([pd.read_csv(f) for f in tcga_files], ignore_index=True)
    gene_cols = [c for c in tcga_expr.columns
                 if c not in ("sample_id", "cohort", "cancer_type")]

    model = pretrain_on_tcga(tcga_expr, gene_cols)

    if args.target_cohort:
        target_expr = pd.read_csv(args.target_cohort)
        scores = apply_pretrained_score(model, target_expr, gene_cols)
        out_df = pd.DataFrame({"sample_id": target_expr["sample_id"],
                                "tcga_immune_activity_score": scores.values})
        out_df.to_csv(args.output, index=False)
        print(f"[ok] wrote {args.output}")
    else:
        print("[info] no --target-cohort given; pretrained model not applied to any data. "
              "Re-run with --target-cohort to generate the engineered feature.")


if __name__ == "__main__":
    main()
