"""
src/inference.py

Shared logic for turning a trained model into something reusable: saving
the fitted model + the exact feature list + preprocessing parameters to
disk, and applying that same preprocessing to a brand-new sample at
prediction time.

IMPORTANT LIMITATION, stated up front rather than hidden in a comment
somewhere: this project's training data (merged_expression.csv) has
already been through ComBat batch correction across the 3 training
cohorts. ComBat is a multi-sample method -- it does not have a
well-defined way to "correct" a single new sample from an unknown batch.
This module does NOT attempt to re-run ComBat on new samples. Instead it:
  1. log2(x+1)-transforms the new sample's raw values (same as training)
  2. imputes any of the selected genes missing from the new sample using
     that gene's TRAINING-SET median (saved at model-fitting time)
  3. feeds the result directly into the trained model

This means predictions on a new sample from a very different sequencing
platform/batch than the training cohorts should be treated with real
caution -- there is no guarantee the new sample's values sit on the same
scale as what the model was trained on. This is documented, not solved;
see the README section this module is referenced from for the honest
discussion of what would be needed to solve it properly (e.g. reference-
based normalization, or refitting ComBat with the new sample included as
its own batch if you have enough new samples to do so meaningfully).
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
import shap


def save_model_artifacts(model, selected_features: list, training_medians: pd.Series,
                          out_dir: str, model_name: str = "final_model"):
    """
    Save everything needed to reproduce predictions later:
      - the fitted sklearn/xgboost model (joblib)
      - the exact ordered list of feature (gene) names it expects
      - each feature's training-set median, for imputing missing genes
        in a new sample
    """
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(model, os.path.join(out_dir, f"{model_name}.joblib"))

    with open(os.path.join(out_dir, f"{model_name}_features.json"), "w") as f:
        json.dump(selected_features, f, indent=2)

    training_medians.loc[selected_features].to_csv(
        os.path.join(out_dir, f"{model_name}_medians.csv"), header=["median"]
    )

    print(f"[ok] saved model artifacts to {out_dir}/:")
    print(f"  {model_name}.joblib          -- the fitted model")
    print(f"  {model_name}_features.json   -- {len(selected_features)} expected gene names, in order")
    print(f"  {model_name}_medians.csv     -- training medians, used to impute missing genes")


def load_model_artifacts(model_dir: str, model_name: str = "final_model"):
    """Load everything saved by save_model_artifacts()."""
    model = joblib.load(os.path.join(model_dir, f"{model_name}.joblib"))

    with open(os.path.join(model_dir, f"{model_name}_features.json")) as f:
        selected_features = json.load(f)

    medians = pd.read_csv(os.path.join(model_dir, f"{model_name}_medians.csv"), index_col=0)["median"]

    return model, selected_features, medians


def preprocess_new_sample(raw_expression: pd.DataFrame, selected_features: list,
                           training_medians: pd.Series, already_log_transformed: bool = False):
    """
    Prepare a new sample (or batch of new samples) for prediction.

    raw_expression: DataFrame with sample_id as one column and gene
        expression values as the other columns (same convention as the
        rest of this project -- see e.g. data/raw/*_expression.csv).
    already_log_transformed: set True if your input is already
        log2(TPM+1) or similar (e.g. if it came from the same upstream
        pipeline as the training cohorts). Leave False for raw
        counts/TPM values, which is the more common case for a
        genuinely new sample.

    Returns a DataFrame with exactly `selected_features` as columns, in
    the same order the model expects, with missing genes imputed from
    training_medians and a warning printed for each one.
    """
    df = raw_expression.copy()
    sample_ids = df["sample_id"] if "sample_id" in df.columns else df.index

    gene_cols = [c for c in df.columns if c != "sample_id"]
    numeric = df[gene_cols].apply(pd.to_numeric, errors="coerce")

    if not already_log_transformed:
        numeric = np.log2(numeric.clip(lower=0) + 1)

    missing_genes = [g for g in selected_features if g not in numeric.columns]
    if missing_genes:
        print(f"[warn] {len(missing_genes)} of {len(selected_features)} expected genes "
              f"are missing from this input and will be imputed with training medians: "
              f"{missing_genes[:10]}{'...' if len(missing_genes) > 10 else ''}")
        for g in missing_genes:
            numeric[g] = training_medians.get(g, 0.0)

    numeric = numeric[selected_features]  # exact order the model expects
    numeric = numeric.fillna(training_medians.loc[selected_features])

    numeric.insert(0, "sample_id", sample_ids.values)
    return numeric


def predict(model, prepared_df: pd.DataFrame, selected_features: list) -> pd.DataFrame:
    """Run the model on a preprocessed DataFrame, return predictions + probabilities."""
    X = prepared_df[selected_features]
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = np.where(y_prob >= 0.5, "responder", "non_responder")

    return pd.DataFrame({
        "sample_id": prepared_df["sample_id"].values,
        "predicted_response": y_pred,
        "responder_probability": y_prob,
    })


def explain_predictions(model, prepared_df: pd.DataFrame, selected_features: list,
                         training_medians: pd.Series, top_n: int = 10) -> dict:
    """
    Per-sample SHAP explanation for why the model predicted what it did.

    Only supports the StandardScaler -> LogisticRegression pipeline produced by
    train_final_model.py's --model logreg (the deployed default). Uses
    shap.LinearExplainer, which is exact (not an approximation) for linear
    models -- the baseline + sum of per-feature contributions reconstructs
    the model's raw logit exactly.

    Baseline: the training-set median for every feature (i.e. "what a
    typical training patient looks like"). Each feature's SHAP value says
    how much that gene's value in THIS sample pushed the prediction away
    from that typical baseline, in log-odds units. Positive = pushed toward
    "responder", negative = pushed toward "non_responder".

    Returns a dict mapping sample_id -> DataFrame with columns:
        gene, shap_value, direction, sample_value_log2, training_median_log2
    sorted by absolute contribution, top_n rows only.

    Raises TypeError if the model isn't the expected Pipeline(prep, clf) shape
    (e.g. --model rf/svm/xgboost/ensemble) -- those need a different SHAP
    explainer (TreeExplainer, KernelExplainer) not implemented here.
    """
    try:
        prep = model.named_steps["prep"]
        clf = model.named_steps["clf"]
    except (AttributeError, KeyError) as e:
        raise TypeError(
            "explain_predictions only supports the logreg Pipeline(prep, clf) model. "
            "SHAP explanations for other model types (rf/svm/xgboost/ensemble) are "
            "not implemented in this module."
        ) from e

    X = prepared_df[selected_features]
    X_scaled = prep.transform(X)

    background_raw = training_medians.loc[selected_features].values.reshape(1, -1)
    background_scaled = prep.transform(pd.DataFrame(background_raw, columns=selected_features))
    masker = shap.maskers.Independent(background_scaled)
    explainer = shap.LinearExplainer(clf, masker)
    shap_values = explainer(X_scaled)

    results = {}
    for i, sample_id in enumerate(prepared_df["sample_id"].values):
        sv = shap_values.values[i]
        contrib = pd.DataFrame({
            "gene": selected_features,
            "shap_value": sv,
            "sample_value_log2": X.iloc[i].values,
            "training_median_log2": training_medians.loc[selected_features].values,
        })
        contrib["direction"] = np.where(
            contrib["shap_value"] > 0, "toward responder", "toward non-responder"
        )
        contrib["abs_shap"] = contrib["shap_value"].abs()
        top = contrib.sort_values("abs_shap", ascending=False).head(top_n)
        results[sample_id] = top.drop(columns="abs_shap").reset_index(drop=True)

    return results
