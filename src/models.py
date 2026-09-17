"""
Phase 5: Train and compare classifiers (LR, RF, SVM, XGBoost, and a soft-
voting ensemble), following the evaluation practice from Tran/Waddell 2026:
repeated stratified k-fold CV, AUC-ROC + weighted F1 as primary metrics,
Youden-index-based optimal threshold, isotonic calibration + Brier score.
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (roc_auc_score, f1_score, precision_score, recall_score,
                              confusion_matrix, brier_score_loss, roc_curve)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False


def build_preprocessor(numeric_cols: list, categorical_cols: list) -> ColumnTransformer:
    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))
    return ColumnTransformer(transformers)


def build_models(preprocessor: ColumnTransformer, random_state: int = 42) -> dict:
    lr = Pipeline([("prep", preprocessor),
                   ("clf", LogisticRegression(max_iter=2000, random_state=random_state))])
    rf = Pipeline([("prep", preprocessor),
                   ("clf", RandomForestClassifier(n_estimators=500, random_state=random_state, n_jobs=-1))])
    svm = Pipeline([("prep", preprocessor),
                    ("clf", SVC(probability=True, random_state=random_state))])

    models = {"LogisticRegression": lr, "RandomForest": rf, "SVM": svm}

    if HAS_XGB:
        xgb = Pipeline([("prep", preprocessor),
                        ("clf", XGBClassifier(n_estimators=300, use_label_encoder=False,
                                               eval_metric="logloss", random_state=random_state))])
        models["XGBoost"] = xgb

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf)], voting="soft"
    )
    models["Ensemble_LR_RF"] = ensemble

    return models


def youden_optimal_threshold(y_true, y_prob) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j = tpr - fpr
    return float(thresholds[np.argmax(j)])


def evaluate_model_cv(model, X, y, n_splits: int = 5, n_repeats: int = 10,
                       random_state: int = 42) -> dict:
    """Repeated stratified k-fold CV, reporting AUC-ROC and weighted F1."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1)
    return {
        "auc_roc_mean": float(np.mean(auc_scores)),
        "auc_roc_std": float(np.std(auc_scores)),
        "f1_weighted_mean": float(np.mean(f1_scores)),
        "f1_weighted_std": float(np.std(f1_scores)),
    }


def evaluate_on_holdout(model, X_train, y_train, X_test, y_test) -> dict:
    """Fit on train, evaluate on a genuinely held-out test set."""
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    threshold = youden_optimal_threshold(y_train, model.predict_proba(X_train)[:, 1])
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_test, y_pred).tolist()
    return {
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": cm,  # [[TN, FP], [FN, TP]]
        "youden_threshold": threshold,
    }


def calibrate_and_score(model, X_train, y_train, X_test, y_test, method: str = "isotonic") -> dict:
    """Isotonic calibration + Brier score, as in the reference paper."""
    calibrated = CalibratedClassifierCV(model, method=method, cv=5)
    calibrated.fit(X_train, y_train)
    y_prob = calibrated.predict_proba(X_test)[:, 1]
    return {"brier_score": float(brier_score_loss(y_test, y_prob))}


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate ICI response classifiers.")
    parser.add_argument("--data", required=True, help="Feature table CSV from feature_selection.py")
    parser.add_argument("--output", required=True, help="JSON path for results")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    meta_cols = {"sample_id", "cohort", "cancer_type", "response_binary"}
    feature_cols = [c for c in df.columns if c not in meta_cols]
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    from sklearn.model_selection import train_test_split
    y = (df["response_binary"] == "responder").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols], y, test_size=args.test_size, stratify=y, random_state=42
    )

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    models = build_models(preprocessor)

    results = {}
    for name, model in models.items():
        print(f"[info] evaluating {name} ...")
        cv_scores = evaluate_model_cv(model, X_train, y_train)
        holdout_scores = evaluate_on_holdout(model, X_train, y_train, X_test, y_test)
        results[name] = {"cross_validation": cv_scores, "holdout_test": holdout_scores}

    best_model_name = max(results, key=lambda n: results[n]["holdout_test"]["auc_roc"])
    print(f"[info] best model on holdout AUC-ROC: {best_model_name}")

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[ok] wrote {args.output}")


if __name__ == "__main__":
    main()
