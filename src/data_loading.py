"""
Phase 1: Data loading and harmonization.

Public ICI cohorts come in different formats (raw counts, TPM, FPKM;
different gene ID systems; different response-label conventions). This
module standardizes them into one common schema:

    sample_id | cancer_type | cohort | gene_1 ... gene_N | response

`response` is harmonized to a binary label:
    CR / PR              -> "responder"
    SD / PD               -> "non_responder"
(exact cutoffs should be reconsidered per cohort's own clinical annotation --
 see the paper's discussion of why stable disease is ambiguous.)

NOTE: This script does not download proprietary/controlled-access data for
you (e.g. dbGaP cohorts need an approved data access request). It gives you
the harmonization logic to apply once you have the files locally.
"""
import argparse
import os
import pandas as pd


RESPONSE_MAP = {
    "CR": "responder",
    "PR": "responder",
    "SD": "non_responder",   # reconsider per-cohort; some studies treat SD as mixed
    "PD": "non_responder",
}


def harmonize_response_labels(df: pd.DataFrame, response_col: str = "response") -> pd.DataFrame:
    """
    Map raw RECIST-style labels (CR/PR/SD/PD) to responder/non_responder.

    If the file already has a 'response_binary' column (e.g. produced by a
    cohort-specific prep script like scripts/prepare_hugo2016.py, which maps
    labels from the cohort's own vocabulary -- "Complete Response" rather
    than "CR", etc.), that column is trusted as-is and this step is skipped,
    rather than looking for a 'response' column that won't exist.
    """
    df = df.copy()
    if "response_binary" in df.columns:
        n_missing = df["response_binary"].isna().sum()
        if n_missing:
            print(f"[warn] {n_missing} samples have a missing response_binary "
                  f"value and were dropped.")
        return df.dropna(subset=["response_binary"])

    if response_col not in df.columns:
        raise KeyError(
            f"Neither 'response_binary' nor '{response_col}' found in columns "
            f"{list(df.columns)}. Either pre-compute response_binary (see "
            f"scripts/prepare_hugo2016.py) or pass the correct --response_col."
        )
    df["response_binary"] = df[response_col].map(RESPONSE_MAP)
    n_dropped = df["response_binary"].isna().sum()
    if n_dropped:
        print(f"[warn] {n_dropped} samples had unmapped response labels and were dropped.")
    return df.dropna(subset=["response_binary"])


def load_expression_matrix(path: str, sample_id_col: str = "sample_id") -> pd.DataFrame:
    """
    Load a gene expression matrix (genes as columns, samples as rows, or vice
    versa -- this function expects samples as rows). CSV/TSV auto-detected.
    """
    sep = "\t" if path.endswith(".tsv") or path.endswith(".txt") else ","
    df = pd.read_csv(path, sep=sep)
    if sample_id_col not in df.columns:
        raise ValueError(
            f"Expected a '{sample_id_col}' column. Found: {list(df.columns)[:10]}..."
        )
    return df


def load_clinical_metadata(path: str) -> pd.DataFrame:
    """Load clinical/response metadata (age, sex, response label, etc.)."""
    sep = "\t" if path.endswith(".tsv") else ","
    return pd.read_csv(path, sep=sep)


def merge_cohorts(expression_dfs: list, cohort_names: list, cancer_types: list) -> pd.DataFrame:
    """
    Concatenate multiple cohorts into one long table, tagging each row with
    its cohort and cancer type -- required for later leave-one-cohort-out
    and cross-cancer-type validation.
    """
    assert len(expression_dfs) == len(cohort_names) == len(cancer_types)
    tagged = []
    for df, cohort, cancer in zip(expression_dfs, cohort_names, cancer_types):
        df = df.copy()
        df["cohort"] = cohort
        df["cancer_type"] = cancer
        tagged.append(df)
    merged = pd.concat(tagged, axis=0, ignore_index=True, join="outer")
    print(f"[info] merged {len(tagged)} cohorts -> {merged.shape[0]} samples, "
          f"{merged.shape[1]} columns")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Load and tag a single cohort.")
    parser.add_argument("--cohort", required=True, help="Cohort name, e.g. hugo2016")
    parser.add_argument("--cancer", required=True, help="Cancer type, e.g. melanoma")
    parser.add_argument("--expression", help="Path to expression matrix (CSV/TSV)")
    parser.add_argument("--clinical", help="Path to clinical metadata (CSV/TSV)")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    def add_cohort_tags(df: pd.DataFrame) -> pd.DataFrame:
        """Add cohort/cancer_type columns only if not already present (a
        cohort-specific prep script like prepare_hugo2016.py may already
        have set them -- avoid duplicate columns)."""
        missing = {}
        if "cohort" not in df.columns:
            missing["cohort"] = [args.cohort] * len(df)
        if "cancer_type" not in df.columns:
            missing["cancer_type"] = [args.cancer] * len(df)
        if missing:
            df = pd.concat([df, pd.DataFrame(missing, index=df.index)], axis=1).copy()
        return df

    if args.expression:
        expr = load_expression_matrix(args.expression)
        expr = add_cohort_tags(expr)
        out_path = os.path.join(args.out, f"{args.cohort}_expression.csv")
        expr.to_csv(out_path, index=False)
        print(f"[ok] wrote {out_path}")

    if args.clinical:
        clin = load_clinical_metadata(args.clinical)
        clin = harmonize_response_labels(clin)
        clin = add_cohort_tags(clin)
        out_path = os.path.join(args.out, f"{args.cohort}_clinical.csv")
        clin.to_csv(out_path, index=False)
        print(f"[ok] wrote {out_path}")

    if not args.expression and not args.clinical:
        print("[info] No --expression or --clinical path given. This cohort "
              "still needs to be downloaded manually -- see README Phase 1 "
              "table for the accession number, then re-run with the local paths.")


if __name__ == "__main__":
    main()
