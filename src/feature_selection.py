"""
Phase 4: Feature selection, following the Tran/Waddell 2026 recipe:
  1. Univariate Mann-Whitney U screen (responder vs non-responder), p <= 0.05
  2. Hierarchical clustering on pairwise correlation to find collinear groups
  3. Within each collinear cluster, keep the single feature with the
     highest median absolute SHAP score (computed after an initial model fit)
"""
import argparse
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.ensemble import RandomForestClassifier
import shap


def _extract_positive_class_shap(shap_values) -> np.ndarray:
    """
    Normalize SHAP output across shap library versions/explainer types into
    a plain (n_samples, n_features) array of SHAP values for the positive
    (responder) class:
      - list [class0_array, class1_array]           -> take class1_array
      - 3D array (n_samples, n_features, n_classes)  -> take [..., -1]
      - 2D array (n_samples, n_features)             -> already correct
    """
    if isinstance(shap_values, list):
        return np.asarray(shap_values[-1])
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, -1]
    return arr


def univariate_screen(df: pd.DataFrame, feature_cols: list, label_col: str = "response_binary",
                       alpha: float = 0.05) -> list:
    """Mann-Whitney U test per feature; keep those with p <= alpha."""
    responders = df[df[label_col] == "responder"]
    non_responders = df[df[label_col] == "non_responder"]

    numeric_feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_feature_cols = [c for c in feature_cols if c not in numeric_feature_cols]

    kept = list(categorical_feature_cols)  # keep categorical features automatically (e.g. sex, mutation status)
    for col in numeric_feature_cols:
        a = responders[col].dropna()
        b = non_responders[col].dropna()
        if len(a) < 3 or len(b) < 3:
            continue
        try:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
        except ValueError:
            continue
        if p <= alpha:
            kept.append(col)

    print(f"[info] univariate screen: {len(kept)}/{len(feature_cols)} features passed p<={alpha}")
    return kept


def cluster_collinear_features(df: pd.DataFrame, feature_cols: list,
                                corr_threshold: float = 0.7) -> dict:
    """
    Hierarchical clustering on 1 - |Pearson r| distance. Returns
    {cluster_id: [feature_names]}.
    """
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    corr = df[numeric_cols].corr().abs().fillna(0)
    dist = 1 - corr
    condensed = squareform(dist.values, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=1 - corr_threshold, criterion="distance")

    clusters = {}
    for feature, cid in zip(numeric_cols, cluster_ids):
        clusters.setdefault(cid, []).append(feature)
    return clusters


def select_representative_by_shap(df: pd.DataFrame, clusters: dict, label_col: str = "response_binary",
                                   random_state: int = 42) -> list:
    """
    Fit a quick random forest, compute SHAP values, and within each collinear
    cluster keep only the feature with the highest median |SHAP| score.
    Singleton clusters are kept automatically.
    """
    all_features = [f for cluster in clusters.values() for f in cluster]
    X_raw = df[all_features].copy()
    numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in all_features if c not in numeric_cols]

    X_raw[numeric_cols] = X_raw[numeric_cols].fillna(X_raw[numeric_cols].median())
    X = pd.get_dummies(X_raw, columns=categorical_cols, drop_first=False)
    y = (df[label_col] == "responder").astype(int)

    model = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1)
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    sv = _extract_positive_class_shap(shap_values)
    median_abs_shap = pd.Series(np.median(np.abs(sv), axis=0), index=X.columns)

    selected = []
    for cid, feats in clusters.items():
        if len(feats) == 1:
            selected.append(feats[0])
        else:
            best = median_abs_shap.loc[feats].idxmax()
            selected.append(best)

    print(f"[info] SHAP-based collinearity pruning: {len(selected)} features kept "
          f"from {len(all_features)} across {len(clusters)} clusters")
    return selected, median_abs_shap


def run_feature_selection(df: pd.DataFrame, feature_cols: list, label_col: str = "response_binary",
                           alpha: float = 0.05, corr_threshold: float = 0.7,
                           max_features: int = 100) -> list:
    """
    max_features: hard cap on the final feature count, applied AFTER
    collinearity pruning, keeping the highest-|SHAP| features. This is
    essential when the number of samples is small (e.g. ~150-250, as in
    this project's real cohorts) -- without a cap, collinearity pruning
    alone can still leave thousands of near-independent features, giving a
    features-to-samples ratio that overfits badly. A symptom of this is
    every model predicting a single class on the holdout set despite
    reasonable-looking AUC (AUC measures ranking, not the usability of the
    fitted decision threshold). As a rule of thumb, aim for at least
    5-10 training samples per feature.
    """
    screened = univariate_screen(df, feature_cols, label_col, alpha)
    if len(screened) < 2:
        print("[warn] fewer than 2 features passed screening; skipping collinearity step")
        return screened

    numeric_screened = df[screened].select_dtypes(include=[np.number]).columns.tolist()
    categorical_screened = [c for c in screened if c not in numeric_screened]

    clusters = cluster_collinear_features(df, numeric_screened, corr_threshold)
    # add categorical features as their own singleton clusters so they survive
    # the SHAP-based pruning step below untouched
    next_cluster_id = max(clusters.keys(), default=0) + 1
    for i, cat_col in enumerate(categorical_screened):
        clusters[next_cluster_id + i] = [cat_col]

    final, shap_importance = select_representative_by_shap(df, clusters, label_col)

    if len(final) > max_features:
        final = shap_importance.loc[final].sort_values(ascending=False).head(max_features).index.tolist()
        print(f"[info] capped feature set at max_features={max_features} "
              f"(kept the top {max_features} by median |SHAP|)")

    return final


def main():
    parser = argparse.ArgumentParser(description="Select final feature set (MWU + SHAP collinearity pruning).")
    parser.add_argument("--features", required=True)
    parser.add_argument("--network-features", default=None)
    parser.add_argument("--labels", default=None, help="Optional separate labels CSV (sample_id, response_binary)")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--corr-threshold", type=float, default=0.7)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.features)
    if args.network_features:
        net = pd.read_csv(args.network_features)
        df = df.merge(net, on="sample_id", how="left")
    if args.labels:
        labels = pd.read_csv(args.labels)
        df = df.merge(labels, on="sample_id", how="left")

    meta_cols = {"sample_id", "cohort", "cancer_type", "response_binary", "response"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    selected = run_feature_selection(df, feature_cols, alpha=args.alpha,
                                      corr_threshold=args.corr_threshold)

    out_cols = ["sample_id", "cohort", "cancer_type", "response_binary"] + selected
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_csv(args.output, index=False)
    print(f"[ok] wrote {args.output} with {len(selected)} selected features")


if __name__ == "__main__":
    main()
