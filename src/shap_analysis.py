"""
Phase 7a: SHAP explainability -- global feature importance and per-feature
numeric threshold discovery, following the LOWESS-intersection method from
Tran/Waddell 2026: fit a smoothed curve of SHAP score vs. feature value
separately for responders and non-responders, and find where the two curves
cross (the value at which a feature flips from "supports response" to
"opposes response").
"""
import argparse
import os
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from statsmodels.nonparametric.smoothers_lowess import lowess

from src.feature_selection import _extract_positive_class_shap


def fit_and_explain(df: pd.DataFrame, feature_cols: list, label_col: str = "response_binary",
                     random_state: int = 42):
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(
        df[feature_cols].select_dtypes(include=[np.number]).median()
    )
    y = (df[label_col] == "responder").astype(int)

    model = RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    sv = _extract_positive_class_shap(shap_values)

    return model, X, y, sv


def global_feature_importance(X: pd.DataFrame, shap_values: np.ndarray) -> pd.DataFrame:
    """Median absolute SHAP score per feature, sorted descending."""
    median_abs = np.median(np.abs(shap_values), axis=0)
    out = pd.DataFrame({"feature": X.columns, "median_abs_shap": median_abs})
    return out.sort_values("median_abs_shap", ascending=False).reset_index(drop=True)


def find_threshold_via_lowess(feature_values: np.ndarray, shap_col: np.ndarray,
                               labels: np.ndarray, frac: float = 0.6):
    """
    Fit LOWESS curves of SHAP score vs. feature value separately for
    responders (label=1) and non-responders (label=0), then find the
    feature value where the two curves intersect -- the candidate threshold.

    Returns None if the curves don't cross within the observed value range
    (i.e. no clean threshold could be identified for this feature).
    """
    order = np.argsort(feature_values)
    fv, sc, lb = feature_values[order], shap_col[order], labels[order]

    resp_mask, non_resp_mask = lb == 1, lb == 0
    if resp_mask.sum() < 5 or non_resp_mask.sum() < 5:
        return None  # not enough points to fit a stable LOWESS curve

    resp_smooth = lowess(sc[resp_mask], fv[resp_mask], frac=frac, return_sorted=True)
    non_resp_smooth = lowess(sc[non_resp_mask], fv[non_resp_mask], frac=frac, return_sorted=True)

    # Interpolate both curves onto a common grid, then find sign changes
    grid = np.linspace(max(resp_smooth[0, 0], non_resp_smooth[0, 0]),
                        min(resp_smooth[-1, 0], non_resp_smooth[-1, 0]), 200)
    if len(grid) < 2:
        return None

    resp_interp = np.interp(grid, resp_smooth[:, 0], resp_smooth[:, 1])
    non_resp_interp = np.interp(grid, non_resp_smooth[:, 0], non_resp_smooth[:, 1])
    diff = resp_interp - non_resp_interp

    sign_changes = np.where(np.diff(np.sign(diff)))[0]
    if len(sign_changes) == 0:
        return None

    # Return the first crossing point (in original feature-value units)
    return float(grid[sign_changes[0]])


def discover_thresholds(X: pd.DataFrame, y: np.ndarray, shap_values: np.ndarray,
                         top_n: int = 5, frac: float = 0.85) -> pd.DataFrame:
    """
    Run threshold discovery for the top_n most important features.

    Defaults changed from top_n=10/frac=0.6 to top_n=5/frac=0.85: with small
    per-cancer-type sample sizes (as low as 16 in this project), a narrow
    LOWESS window has too few points in some regions of a feature's value
    range to fit a stable curve, so most features returned no threshold at
    all. Fewer, more important features + a wider smoothing window trades
    resolution for a much higher chance of actually resolving a threshold.
    """
    importance = global_feature_importance(X, shap_values)
    top_features = importance.head(top_n)["feature"].tolist()

    rows = []
    for i, feat in enumerate(X.columns):
        if feat not in top_features:
            continue
        threshold = find_threshold_via_lowess(X[feat].values, shap_values[:, i],
                                               y.values if hasattr(y, "values") else y,
                                               frac=frac)
        rows.append({"feature": feat, "threshold": threshold})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="SHAP importance + threshold discovery.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--per-cancer-type", action="store_true",
                         help="Run threshold discovery separately per cancer_type "
                              "(needed for Phase 7b cross-cancer threshold comparison)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    df = pd.read_csv(args.data)
    meta_cols = {"sample_id", "cohort", "cancer_type", "response_binary"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    if args.per_cancer_type:
        all_thresholds = []
        for cancer_type in df["cancer_type"].unique():
            sub = df[df["cancer_type"] == cancer_type]
            if sub["response_binary"].nunique() < 2 or len(sub) < 15:
                print(f"[warn] skipping {cancer_type}: too few samples/classes")
                continue
            model, X, y, sv = fit_and_explain(sub, feature_cols)
            importance = global_feature_importance(X, sv)
            importance["cancer_type"] = cancer_type
            importance.to_csv(os.path.join(args.output, f"importance_{cancer_type}.csv"), index=False)

            thresholds = discover_thresholds(X, y, sv)
            thresholds["cancer_type"] = cancer_type
            all_thresholds.append(thresholds)

        pd.concat(all_thresholds, ignore_index=True).to_csv(
            os.path.join(args.output, "thresholds_by_cancer_type.csv"), index=False
        )
    else:
        model, X, y, sv = fit_and_explain(df, feature_cols)
        importance = global_feature_importance(X, sv)
        importance.to_csv(os.path.join(args.output, "importance_overall.csv"), index=False)

        thresholds = discover_thresholds(X, y, sv)
        thresholds.to_csv(os.path.join(args.output, "thresholds_overall.csv"), index=False)

    print(f"[ok] wrote SHAP results to {args.output}")


if __name__ == "__main__":
    main()
