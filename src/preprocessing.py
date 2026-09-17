"""
Phase 1 (continued): QC, normalization, missing-value handling, batch
correction, train/test split.
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def remove_low_quality_samples(df: pd.DataFrame, gene_cols: list,
                                 max_missing_frac: float = 0.3) -> pd.DataFrame:
    """Drop samples with too much missing expression data."""
    missing_frac = df[gene_cols].isna().mean(axis=1)
    keep = missing_frac <= max_missing_frac
    n_dropped = (~keep).sum()
    if n_dropped:
        print(f"[warn] dropping {n_dropped} low-quality samples "
              f"(>{max_missing_frac:.0%} missing genes)")
    return df.loc[keep].reset_index(drop=True)


def remove_low_variance_genes(df: pd.DataFrame, gene_cols: list,
                               min_variance: float = 1e-6) -> list:
    """Return the subset of gene_cols that have non-trivial variance."""
    variances = df[gene_cols].var(axis=0, skipna=True)
    kept = variances[variances > min_variance].index.tolist()
    print(f"[info] kept {len(kept)}/{len(gene_cols)} genes after variance filter")
    return kept


def impute_missing(df: pd.DataFrame, gene_cols: list) -> pd.DataFrame:
    """Median-impute missing gene expression values (simple, robust default)."""
    df = df.copy()
    df[gene_cols] = df[gene_cols].fillna(df[gene_cols].median())
    return df


def log_transform(df: pd.DataFrame, gene_cols: list) -> pd.DataFrame:
    """log2(TPM + 1) transform, standard for RNA-seq expression."""
    df = df.copy()
    df[gene_cols] = np.log2(df[gene_cols].clip(lower=0) + 1)
    return df


def batch_correct_combat(df: pd.DataFrame, gene_cols: list, batch_col: str = "cohort") -> pd.DataFrame:
    """
    ComBat batch correction across cohorts. Requires the `combat` package
    (pip install combat) or use pycombat from the `inmoose` package.
    Falls back to a simple per-batch mean-centering if combat isn't
    installed -- adequate for a first pass, but ComBat is recommended for
    anything you plan to publish.
    """
    try:
        from combat.pycombat import pycombat
        expr_t = df[gene_cols].T  # genes as rows, samples as columns (pycombat convention)
        corrected = pycombat(expr_t, df[batch_col])
        df = df.copy()
        df[gene_cols] = corrected.T.values
        print("[info] applied ComBat batch correction")
    except ImportError:
        print("[warn] `combat` package not installed -- falling back to "
              "per-batch mean-centering. Run `pip install combat` for proper "
              "ComBat correction before publishing results.")
        df = df.copy()
        for batch in df[batch_col].unique():
            mask = df[batch_col] == batch
            df.loc[mask, gene_cols] = (
                df.loc[mask, gene_cols] - df.loc[mask, gene_cols].mean()
            )
    return df


def encode_clinical_variables(df: pd.DataFrame, categorical_cols: list) -> pd.DataFrame:
    """One-hot encode categorical clinical variables (sex, mutation status, etc.)."""
    return pd.get_dummies(df, columns=[c for c in categorical_cols if c in df.columns],
                           drop_first=False)


def run_preprocessing_pipeline(df: pd.DataFrame, gene_cols: list,
                                categorical_cols: list = None,
                                batch_correct: bool = True) -> pd.DataFrame:
    """Full Phase-1 pipeline in the order recommended in the project README."""
    categorical_cols = categorical_cols or []
    df = remove_low_quality_samples(df, gene_cols)
    gene_cols = remove_low_variance_genes(df, gene_cols)
    df = impute_missing(df, gene_cols)
    df = log_transform(df, gene_cols)
    if batch_correct and df["cohort"].nunique() > 1:
        df = batch_correct_combat(df, gene_cols)
    df = encode_clinical_variables(df, categorical_cols)
    return df, gene_cols


def split_train_test(df: pd.DataFrame, label_col: str = "response_binary",
                      test_size: float = 0.2, random_state: int = 42):
    """Stratified split, preserving class balance."""
    return train_test_split(df, test_size=test_size, stratify=df[label_col],
                             random_state=random_state)


def main():
    parser = argparse.ArgumentParser(description="Merge and preprocess cohort files.")
    parser.add_argument("--input", required=True, help="Directory of per-cohort CSVs from data_loading.py")
    parser.add_argument("--output", required=True, help="Output path for merged, cleaned CSV")
    parser.add_argument("--no-batch-correct", action="store_true")
    args = parser.parse_args()

    expr_files = glob.glob(os.path.join(args.input, "*_expression.csv"))
    if not expr_files:
        raise FileNotFoundError(f"No *_expression.csv files found in {args.input}")

    dfs = []
    per_cohort_gene_cols = []  # list of sets, one per cohort's expression file

    for expr_path in expr_files:
        expr = pd.read_csv(expr_path)

        non_gene_tags = {"sample_id", "cohort", "cancer_type"}
        cohort_gene_cols = set(expr.columns) - non_gene_tags
        per_cohort_gene_cols.append(cohort_gene_cols)

        clinical_path = expr_path.replace("_expression.csv", "_clinical.csv")
        if os.path.exists(clinical_path):
            clin = pd.read_csv(clinical_path)
            overlap_cols = (set(expr.columns) & set(clin.columns)) - {"sample_id"}
            clin_to_merge = clin.drop(columns=list(overlap_cols), errors="ignore")
            expr = expr.merge(clin_to_merge, on="sample_id", how="left")
            n_missing_label = expr["response_binary"].isna().sum() if "response_binary" in expr.columns else len(expr)
            if n_missing_label:
                print(f"[warn] {os.path.basename(expr_path)}: {n_missing_label} "
                      f"sample(s) have no matching clinical/response record after merge.")
        else:
            print(f"[warn] no matching clinical file found for {expr_path} "
                  f"(expected {clinical_path}) -- this cohort will have no response labels.")

        dfs.append(expr)

    merged = pd.concat(dfs, axis=0, ignore_index=True, join="outer")

    if "response_binary" not in merged.columns or merged["response_binary"].isna().all():
        raise ValueError(
            "No response_binary labels present after merging expression + "
            "clinical files. Check that each cohort's *_clinical.csv exists "
            "alongside its *_expression.csv and both share the same sample_id "
            "values."
        )

    # IMPORTANT: use the INTERSECTION of gene panels across cohorts, not the
    # union. Different cohorts (e.g. Hugo2016's ~25k genes vs IMvigor210's
    # ~31k genes) rarely share an identical gene panel. Taking the union
    # would make every sample "missing" 30-40%+ of its features purely from
    # panel mismatch -- not a real quality issue -- which would wrongly
    # trigger the low-quality-sample filter below and could drop good
    # samples. Restricting to genes measured in every cohort is both
    # methodologically correct (you can't compare a gene that wasn't
    # measured in one of the cohorts) and avoids that failure mode.
    gene_cols = sorted(set.intersection(*per_cohort_gene_cols)) if len(per_cohort_gene_cols) > 1 \
        else sorted(per_cohort_gene_cols[0])
    non_gene_cols = [c for c in merged.columns if c not in gene_cols]
    print(f"[info] {len(per_cohort_gene_cols)} cohort(s) merged; gene panel sizes: "
          f"{[len(g) for g in per_cohort_gene_cols]}")
    print(f"[info] {len(gene_cols)} genes common to ALL cohorts (intersection) will "
          f"be used; {len(non_gene_cols)} non-gene/clinical columns: {non_gene_cols}")
    if len(per_cohort_gene_cols) > 1 and len(gene_cols) < 0.3 * min(len(g) for g in per_cohort_gene_cols):
        print(f"[warn] the shared gene panel ({len(gene_cols)} genes) is much smaller "
              f"than the individual cohorts' panels -- check gene ID conventions "
              f"match across cohorts (symbol vs Ensembl, case sensitivity, etc.) "
              f"before proceeding, this may indicate a mismatch rather than a "
              f"genuinely small overlap.")

    cleaned, gene_cols = run_preprocessing_pipeline(
        merged, gene_cols, categorical_cols=["sex"],
        batch_correct=not args.no_batch_correct,
    )

    # Keep only sample_id + the intersection gene columns + response_binary
    # (+ cohort/cancer_type for downstream leave-one-cohort/cancer-type-out
    # validation). Drop everything else: non-intersected genes (present in
    # only one cohort's original panel, so meaningless to compare across
    # cohorts) and cohort-specific clinical columns like 'pat_id' (a text
    # ID -- would crash feature_selection.py if left in as a "feature") or
    # 'TMB'/'age_yrs'/'treatment' (present for one cohort, absent for the
    # other, so not usable as a shared cross-cohort feature without
    # explicit imputation, which is out of scope here).
    keep_cols = ["sample_id", "cohort", "cancer_type", "response_binary"] + gene_cols
    dropped_cols = [c for c in cleaned.columns if c not in keep_cols]
    if dropped_cols:
        print(f"[info] dropping {len(dropped_cols)} non-shared/non-numeric columns "
              f"from final output (cohort-specific clinical fields and genes not "
              f"common to all cohorts): {dropped_cols[:10]}"
              f"{'...' if len(dropped_cols) > 10 else ''}")
    cleaned = cleaned[[c for c in keep_cols if c in cleaned.columns]]

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    print(f"[ok] wrote {args.output} ({cleaned.shape[0]} samples, {len(gene_cols)} genes)")


if __name__ == "__main__":
    main()
