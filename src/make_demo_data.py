"""
Generates small synthetic multi-cohort, multi-cancer-type expression +
clinical data so you can run the ENTIRE pipeline end-to-end before your real
downloads finish. This is for testing the code only -- results are
meaningless (labels are random-ish), it just proves the plumbing works.

Usage:
    python -m src.make_demo_data --output data/processed/demo_features.csv
"""
import argparse
import numpy as np
import pandas as pd

GENES = [
    "PDCD1", "CD274", "CTLA4", "LAG3", "HAVCR2", "TIGIT", "BTLA", "CD28",
    "ICOS", "ICOSLG", "CD80", "CD86", "CD8A", "CD276", "LGALS9", "PDCD1LG2",
    "GZMA", "PRF1", "GZMB", "CXCL9", "CXCL10", "IFNG", "STAT1", "IDO1",
    "HLA-DRA", "HLA-A", "HLA-B", "HLA-C", "B2M", "TAP1", "TAP2",
    "CD3D", "CD3E", "LCK", "ZAP70", "IRF1", "CD68", "CD163", "MRC1", "NOS2", "TNF",
]

COHORTS = {
    "hugo2016": "melanoma",
    "riaz2017": "melanoma",
    "imvigor210": "bladder",
    "gastric_geo": "gastric",
}


def make_demo_dataset(n_per_cohort: int = 40, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    sample_counter = 0

    for cohort, cancer_type in COHORTS.items():
        for _ in range(n_per_cohort):
            sample_counter += 1
            sample_id = f"{cohort}_S{sample_counter:04d}"

            # simulate a "true" latent responder tendency, correlated with
            # a few key checkpoint genes, so models have *something* to learn
            latent = rng.normal(0, 1)
            expr = {g: rng.normal(5, 1.5) for g in GENES}
            expr["PDCD1"] += 0.8 * latent
            expr["CD274"] += 0.6 * latent
            expr["CD8A"] += 0.7 * latent
            expr["GZMB"] += 0.5 * latent

            response_prob = 1 / (1 + np.exp(-latent))
            response = "responder" if rng.random() < response_prob else "non_responder"

            row = {
                "sample_id": sample_id,
                "cohort": cohort,
                "cancer_type": cancer_type,
                "age": int(rng.integers(30, 85)),
                "sex": rng.choice(["male", "female"]),
                "response_binary": response,
            }
            row.update(expr)
            rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-per-cohort", type=int, default=40)
    args = parser.parse_args()

    df = make_demo_dataset(args.n_per_cohort)
    df.to_csv(args.output, index=False)
    print(f"[ok] wrote synthetic demo dataset: {args.output} "
          f"({df.shape[0]} samples across {df['cohort'].nunique()} cohorts, "
          f"{df['cancer_type'].nunique()} cancer types)")


if __name__ == "__main__":
    main()
