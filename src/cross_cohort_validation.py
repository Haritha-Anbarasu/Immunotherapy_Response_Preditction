"""
Phase 6: Leave-one-cohort-out AND leave-one-cancer-type-out validation.

This is the core novelty-generating step (see README). Two experiments:

  1. leave_one_cohort_out(): train on all cohorts but one, test on the held
     out cohort -- reproduces the Tran/Waddell 2026 design (but works for
     any number of cancer types, not just melanoma).

  2. leave_one_cancer_type_out(): train on all cancer types but one, test on
     the held-out cancer type entirely -- this is the specific experiment
     none of the three reference papers ran, and answers "does a model
     trained on melanoma+bladder generalize to gastric cancer?"
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV

from src.models import build_preprocessor, build_models


def _fit_eval(model, X_train, y_train, X_test, y_test) -> dict:
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    result = {
        "n_train": len(y_train),
        "n_test": len(y_test),
        "auc_roc": float(roc_auc_score(y_test, y_prob)) if len(set(y_test)) > 1 else float("nan"),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
    }

    try:
        calibrated = CalibratedClassifierCV(model, method="isotonic", cv=3)
        calibrated.fit(X_train, y_train)
        y_prob_cal = calibrated.predict_proba(X_test)[:, 1]
        result["brier_score"] = float(brier_score_loss(y_test, y_prob_cal))
    except Exception as e:
        result["brier_score"] = None
        result["calibration_error"] = str(e)

    return result


def leave_one_cohort_out(df: pd.DataFrame, feature_cols: list,
                          numeric_cols: list, categorical_cols: list,
                          label_col: str = "response_binary") -> pd.DataFrame:
    """Train on all-but-one cohort, test on the held-out cohort."""
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    models = build_models(preprocessor)

    rows = []
    for cohort in df["cohort"].unique():
        train_df = df[df["cohort"] != cohort]
        test_df = df[df["cohort"] == cohort]
        if test_df[label_col].nunique() < 1 or train_df[label_col].nunique() < 2:
            continue

        y_train = (train_df[label_col] == "responder").astype(int)
        y_test = (test_df[label_col] == "responder").astype(int)

        for name, model in models.items():
            res = _fit_eval(model, train_df[feature_cols], y_train,
                             test_df[feature_cols], y_test)
            res.update({"held_out_cohort": cohort, "model": name,
                        "held_out_cancer_type": test_df["cancer_type"].iloc[0]})
            rows.append(res)

    return pd.DataFrame(rows)


def leave_one_cancer_type_out(df: pd.DataFrame, feature_cols: list,
                               numeric_cols: list, categorical_cols: list,
                               label_col: str = "response_binary") -> pd.DataFrame:
    """
    Train on all-but-one CANCER TYPE, test on the held-out cancer type.
    This is the key novel experiment -- see module docstring.
    """
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    models = build_models(preprocessor)

    rows = []
    for cancer_type in df["cancer_type"].unique():
        train_df = df[df["cancer_type"] != cancer_type]
        test_df = df[df["cancer_type"] == cancer_type]
        if test_df[label_col].nunique() < 1 or train_df[label_col].nunique() < 2:
            continue

        y_train = (train_df[label_col] == "responder").astype(int)
        y_test = (test_df[label_col] == "responder").astype(int)

        for name, model in models.items():
            res = _fit_eval(model, train_df[feature_cols], y_train,
                             test_df[feature_cols], y_test)
            res.update({"held_out_cancer_type": cancer_type, "model": name})
            rows.append(res)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Cross-cohort and cross-cancer-type validation.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    meta_cols = {"sample_id", "cohort", "cancer_type", "response_binary"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    print("[info] running leave-one-cohort-out validation ...")
    cohort_results = leave_one_cohort_out(df, feature_cols, numeric_cols, categorical_cols)

    print("[info] running leave-one-cancer-type-out validation ...")
    cancer_results = leave_one_cancer_type_out(df, feature_cols, numeric_cols, categorical_cols)

    combined = pd.concat(
        [cohort_results.assign(experiment="leave_one_cohort_out"),
         cancer_results.assign(experiment="leave_one_cancer_type_out")],
        ignore_index=True,
    )
    combined.to_csv(args.output, index=False)
    print(f"[ok] wrote {args.output} ({len(combined)} rows)")


if __name__ == "__main__":
    main()
