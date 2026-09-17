"""
Phase 2: Biomarker feature engineering.

Builds the clinically-motivated feature set used in the reference papers:
  - checkpoint receptor gene expression (PD-1/PDCD1, PD-L1/CD274, CTLA4, LAG3, ...)
  - immune gene signatures (IFN-gamma signature, cytolytic score)
  - immune cell fractions (from CIBERSORTx, run separately -- see README)
  - TMB (if mutation calls are available)
"""
import argparse
import numpy as np
import pandas as pd

# Genes used in Tran/Waddell 2026 and standard ICI biomarker literature
CHECKPOINT_GENES = [
    "PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2", "TIGIT",
    "BTLA", "CD28", "ICOS", "ICOSLG", "CD80", "CD86",
    "CD8A", "CD276", "LGALS9", "PDCD1LG2",
]

CYTOLYTIC_SCORE_GENES = ["GZMA", "PRF1"]
CYTOTOXICITY_GENES = ["GZMB", "PRF1"]
CHEMOKINE_GENES = ["CXCL9", "CXCL10"]
IFNG_SIGNATURE_GENES = ["IFNG", "STAT1", "IDO1", "CXCL10", "CXCL9", "HLA-DRA"]


def compute_checkpoint_expression(df: pd.DataFrame) -> pd.DataFrame:
    """Extract available checkpoint gene columns (already log2(TPM+1))."""
    available = [g for g in CHECKPOINT_GENES if g in df.columns]
    missing = set(CHECKPOINT_GENES) - set(available)
    if missing:
        print(f"[warn] checkpoint genes not found in expression matrix: {missing}")
    return df[["sample_id"] + available].copy()


def compute_cytolytic_score(df: pd.DataFrame) -> pd.Series:
    """Geometric mean of GZMA and PRF1 expression (Rooney et al. 2015 definition)."""
    genes = [g for g in CYTOLYTIC_SCORE_GENES if g in df.columns]
    if len(genes) < 2:
        print("[warn] cytolytic score genes incomplete; returning NaN")
        return pd.Series(np.nan, index=df.index)
    vals = df[genes].clip(lower=0) + 0.01  # small offset like reference paper
    return np.exp(np.log(vals).mean(axis=1))


def compute_ifng_signature(df: pd.DataFrame) -> pd.Series:
    """Simple z-score-averaged IFN-gamma signature (proxy for GSVA-based scores)."""
    genes = [g for g in IFNG_SIGNATURE_GENES if g in df.columns]
    if not genes:
        print("[warn] no IFN-gamma signature genes found; returning NaN")
        return pd.Series(np.nan, index=df.index)
    z = (df[genes] - df[genes].mean()) / df[genes].std()
    return z.mean(axis=1)


def compute_tmb(mutation_df: pd.DataFrame, coding_region_mb: float = 30.0) -> pd.DataFrame:
    """
    TMB = non-synonymous mutations per megabase of coding sequence.
    `mutation_df` expected columns: sample_id, variant_classification.
    `coding_region_mb` ~30 Mb is a common WES panel estimate -- replace with
    your actual assay's covered coding region size.
    """
    nonsyn = {"Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Ins",
              "Frame_Shift_Del", "In_Frame_Ins", "In_Frame_Del", "Splice_Site"}
    filtered = mutation_df[mutation_df["variant_classification"].isin(nonsyn)]
    counts = filtered.groupby("sample_id").size().rename("mutation_count")
    tmb = (counts / coding_region_mb).rename("TMB")
    return tmb.reset_index().rename(columns={"index": "sample_id"})


def merge_cibersortx_results(features_df: pd.DataFrame, cibersort_path: str) -> pd.DataFrame:
    """
    CIBERSORTx must be run manually via https://cibersortx.stanford.edu
    (upload the expression matrix, download the results CSV) -- see README
    Phase 2. This just merges the downloaded results back in.

    IMPORTANT: per CIBERSORTx's own usage guidance, run it SEPARATELY for
    each cohort/batch, not on the pooled multi-cohort matrix, then merge
    the per-cohort results here.
    """
    cs = pd.read_csv(cibersort_path)
    id_col = "Mixture" if "Mixture" in cs.columns else "sample_id"
    cs = cs.rename(columns={id_col: "sample_id"})
    drop_cols = [c for c in ["P-value", "Correlation", "RMSE"] if c in cs.columns]
    cs = cs.drop(columns=drop_cols, errors="ignore")
    return features_df.merge(cs, on="sample_id", how="left")


def build_feature_table(expression_path: str, cibersort_path: str = None,
                         mutation_path: str = None) -> pd.DataFrame:
    df = pd.read_csv(expression_path)

    checkpoint = compute_checkpoint_expression(df)
    features = checkpoint.copy()
    features["cytolytic_score"] = compute_cytolytic_score(df)
    features["ifng_signature"] = compute_ifng_signature(df)

    meta_cols = [c for c in ["sample_id", "cohort", "cancer_type", "response_binary",
                              "age", "sex"] if c in df.columns]
    features = df[meta_cols].merge(features, on="sample_id", how="left")

    if cibersort_path:
        features = merge_cibersortx_results(features, cibersort_path)

    if mutation_path:
        mut_df = pd.read_csv(mutation_path)
        tmb = compute_tmb(mut_df)
        features = features.merge(tmb, on="sample_id", how="left")

    return features


def main():
    parser = argparse.ArgumentParser(description="Build clinical/immune biomarker feature table.")
    parser.add_argument("--expression", required=True)
    parser.add_argument("--cibersort", default=None, help="Path to CIBERSORTx_Results.csv")
    parser.add_argument("--mutations", default=None, help="Path to mutation calls CSV")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    features = build_feature_table(args.expression, args.cibersort, args.mutations)
    features.to_csv(args.output, index=False)
    print(f"[ok] wrote {args.output} ({features.shape[0]} samples, {features.shape[1]} features)")


if __name__ == "__main__":
    main()
