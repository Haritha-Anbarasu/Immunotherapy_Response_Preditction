"""
app.py

Public-facing Streamlit web app for the immunotherapy response predictor.
Wraps the exact same logic as scripts/predict_new_sample.py (load_model_artifacts
+ preprocess_new_sample + predict from src/inference.py) behind a browser UI:
upload a CSV, see predictions, download results, and see a per-sample SHAP
explanation of which genes drove the prediction.

This does NOT change the model, the preprocessing, or the science -- it's
purely a UI layer on top of the pipeline that was already verified end-to-end
on the command line.

Run locally:
    streamlit run app.py

Deploy: push this repo to GitHub (including the models/ folder -- it's ~12KB,
small enough to commit) and connect it on https://share.streamlit.io
"""
import os
import sys
import io
import time
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.inference import load_model_artifacts, preprocess_new_sample, predict, explain_predictions

MODEL_DIR = "models"
MODEL_NAME = "final_model_logreg"

# --- Basic public-deployment safeguards -------------------------------------
MAX_FILE_SIZE_MB = 5
MAX_ROWS = 500
MAX_REQUESTS_PER_WINDOW = 20      # per browser session
RATE_LIMIT_WINDOW_SECONDS = 3600  # 1 hour
SHAP_TOP_N = 10
# ------------------------------------------------------------------------------

st.set_page_config(page_title="Immunotherapy Response Predictor", page_icon="🧬", layout="centered")

# --- Visual identity ----------------------------------------------------------
# Palette: cool pale blue-grey base (not the common warm-cream default), deep
# teal as the primary accent, slate-indigo secondary. Responder/non-responder
# colors are intentionally muted rather than pure red/green, consistent with
# a clinical-report tone. Fraunces (serif) carries the headline; Inter (sans)
# carries everything else -- a pairing meant to read as a research instrument,
# not a generic SaaS dashboard.
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Inter:wght@400;500;600&display=swap');

:root {
    --bg-base: #EEF3F6;
    --bg-panel: #FFFFFF;
    --ink: #1C2B33;
    --ink-soft: #4B5C66;
    --accent-teal: #146C6B;
    --accent-indigo: #33415C;
    --positive: #2F8F5B;
    --negative: #A8425C;
    --border: #D7E0E5;
}

.stApp {
    background: linear-gradient(180deg, #EEF3F6 0%, #E4EDF0 100%);
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: var(--ink);
}

.hero-banner {
    position: relative;
    border-radius: 10px;
    padding: 2.75rem 2.25rem 2.25rem 2.25rem;
    margin-bottom: 1.5rem;
    background: linear-gradient(120deg, #123B3B 0%, #1B5654 55%, #146C6B 100%);
    overflow: hidden;
    border: 1px solid #0E2E2E;
}
.hero-banner svg.hero-motif {
    position: absolute;
    top: 0; right: 0;
    height: 100%;
    width: 46%;
    opacity: 0.55;
}
.hero-title {
    position: relative;
    font-family: 'Fraunces', Georgia, serif;
    font-weight: 600;
    font-size: 2.5rem;
    color: #F4F9F8;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.01em;
    line-height: 1.15;
}
.hero-subtitle {
    position: relative;
    font-family: 'Inter', sans-serif;
    font-size: 1.02rem;
    color: #CFE3E0;
    max-width: 30rem;
    line-height: 1.55;
    margin: 0;
}

/* Panels: hairline borders, minimal radius, no soup of drop shadows */
div[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 8px;
    background: var(--bg-panel);
}

div[data-testid="stFileUploader"] {
    border-radius: 8px;
}
div[data-testid="stFileUploaderDropzone"] {
    border: 1.5px dashed #9FB8BD;
    border-radius: 8px;
    background: #FBFDFD;
}

div[data-testid="stAlert"] {
    border-radius: 8px;
    border: 1px solid var(--border);
}

/* Buttons */
.stButton button, .stDownloadButton button {
    background-color: var(--accent-teal);
    color: #FFFFFF;
    border: none;
    border-radius: 6px;
    font-weight: 500;
}
.stButton button:hover, .stDownloadButton button:hover {
    background-color: #0F5352;
    color: #FFFFFF;
}

/* Metric labels in the accent color for cohesion */
div[data-testid="stMetric"] {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
}

h2, h3 {
    font-family: 'Fraunces', serif;
    font-weight: 600;
    color: var(--accent-indigo);
}

hr {
    border-color: var(--border) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

HERO_MOTIF_SVG = (
    '<svg class="hero-motif" viewBox="0 0 420 320" xmlns="http://www.w3.org/2000/svg">'
    '<defs><linearGradient id="strand" x1="0" y1="0" x2="0" y2="1">'
    '<stop offset="0%" stop-color="#EAF6F4" stop-opacity="0.9"/>'
    '<stop offset="100%" stop-color="#EAF6F4" stop-opacity="0.15"/></linearGradient></defs>'
    '<path d="M40,0 C120,60 -20,100 60,160 C140,220 0,260 80,320" fill="none" stroke="url(#strand)" stroke-width="3"/>'
    '<path d="M120,0 C200,60 60,100 140,160 C220,220 80,260 160,320" fill="none" stroke="url(#strand)" stroke-width="3"/>'
    '<path d="M200,0 C280,60 140,100 220,160 C300,220 160,260 240,320" fill="none" stroke="url(#strand)" stroke-width="2.2"/>'
    '<g fill="#EAF6F4" opacity="0.85">'
    '<circle cx="55" cy="28" r="4"/><circle cx="90" cy="95" r="3"/><circle cx="45" cy="150" r="4.5"/>'
    '<circle cx="95" cy="205" r="3"/><circle cx="60" cy="270" r="4"/><circle cx="135" cy="28" r="3.5"/>'
    '<circle cx="170" cy="95" r="4"/><circle cx="125" cy="150" r="3"/><circle cx="175" cy="205" r="4.5"/>'
    '<circle cx="140" cy="270" r="3.5"/><circle cx="215" cy="40" r="3"/><circle cx="250" cy="100" r="3.5"/>'
    '<circle cx="205" cy="155" r="3"/><circle cx="255" cy="210" r="4"/><circle cx="220" cy="270" r="3"/>'
    '</g></svg>'
)

st.markdown(
    f'<div class="hero-banner">{HERO_MOTIF_SVG}'
    '<p class="hero-title">Immunotherapy Response Predictor</p>'
    '<p class="hero-subtitle">Upload gene expression data for one or more samples to get a '
    'predicted responder / non-responder classification, with probability and a per-gene '
    'explanation of what drove that call.</p></div>',
    unsafe_allow_html=True,
)
# ------------------------------------------------------------------------------

st.warning(
    "**Research use only.** This tool is not a validated diagnostic and must not be used to "
    "make or influence any real clinical or treatment decision. Predictions are exploratory, "
    "trained on a limited set of published cohorts, and have not been reviewed or approved by "
    "any regulatory body. Always consult a qualified oncologist for actual treatment decisions.",
    icon="⚠️",
)

with st.expander("Input format instructions", expanded=False):
    st.markdown(
        f"""
        Upload a **CSV file** with:
        - A `sample_id` column (one row per sample, values must be unique)
        - Gene expression columns, one per gene, using standard gene symbol names
          (e.g. `A1BG`, `NKG7`, `PSMD3`, ...)
        - Values should be **raw counts or TPM** (not already log-transformed),
          unless you check "already log2-transformed" below.
        - Numeric expression values only (no text, no negative values).

        You don't need every gene the model uses -- any missing gene is automatically
        filled in with that gene's median value from the training cohorts, and you'll
        see exactly how many genes were missing after you upload.

        **Limits:** files up to {MAX_FILE_SIZE_MB}MB, up to {MAX_ROWS} samples per upload,
        and up to {MAX_REQUESTS_PER_WINDOW} uploads per hour per browser session, to keep
        this public tool usable for everyone.
        """
    )

already_log = st.checkbox(
    "My data is already log2(TPM+1) transformed",
    value=False,
    help="Leave unchecked for raw counts/TPM, which is the more common case.",
)

uploaded_file = st.file_uploader("Upload sample CSV", type=["csv"])


@st.cache_resource
def get_model_artifacts():
    return load_model_artifacts(MODEL_DIR, MODEL_NAME)


def check_rate_limit() -> tuple[bool, int]:
    """Simple in-session sliding-window rate limit. Resets if the browser tab/session
    is closed -- this is a courtesy throttle against accidental abuse, not a security
    control (a determined bad actor can open new sessions)."""
    now = time.time()
    history = st.session_state.get("request_times", [])
    history = [t for t in history if now - t < RATE_LIMIT_WINDOW_SECONDS]
    st.session_state["request_times"] = history
    remaining = MAX_REQUESTS_PER_WINDOW - len(history)
    return remaining > 0, remaining


def record_request():
    st.session_state.setdefault("request_times", []).append(time.time())


def validate_dataframe(raw_df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable validation errors, empty if the file is usable."""
    errors = []

    if raw_df.empty:
        errors.append("The file has no rows.")
        return errors

    if "sample_id" not in raw_df.columns:
        errors.append("Missing required 'sample_id' column.")
        return errors

    if len(raw_df) > MAX_ROWS:
        errors.append(f"File has {len(raw_df)} rows, which exceeds the {MAX_ROWS}-sample limit per upload.")

    if raw_df["sample_id"].isna().any():
        errors.append("Some rows have a blank 'sample_id'.")

    dupes = raw_df["sample_id"][raw_df["sample_id"].duplicated()].unique().tolist()
    if dupes:
        errors.append(f"Duplicate sample_id values found: {dupes[:10]}")

    gene_cols = [c for c in raw_df.columns if c != "sample_id"]
    if not gene_cols:
        errors.append("No gene expression columns found besides 'sample_id'.")
        return errors

    numeric = raw_df[gene_cols].apply(pd.to_numeric, errors="coerce")
    non_numeric_cols = [c for c in gene_cols if numeric[c].isna().sum() > raw_df[c].isna().sum()]
    if non_numeric_cols:
        errors.append(
            f"{len(non_numeric_cols)} column(s) contain non-numeric values that can't be parsed as "
            f"expression data: {non_numeric_cols[:10]}"
        )

    if (numeric.fillna(0) < 0).any().any():
        errors.append("Negative expression values found -- raw counts/TPM should be >= 0.")

    return errors


def run_pipeline(raw_df: pd.DataFrame, already_log_transformed: bool):
    model, selected_features, training_medians = get_model_artifacts()

    gene_cols = [c for c in raw_df.columns if c != "sample_id"]
    present = [g for g in selected_features if g in gene_cols]
    missing = [g for g in selected_features if g not in gene_cols]

    prepared = preprocess_new_sample(
        raw_df, selected_features, training_medians,
        already_log_transformed=already_log_transformed,
    )
    results = predict(model, prepared, selected_features)
    return results, prepared, present, missing, selected_features


def render_explanation(sample_id: str, explanation_df: pd.DataFrame):
    """Render one sample's top SHAP contributors as a horizontal diverging bar chart."""
    plot_df = explanation_df.copy().sort_values("shap_value")
    plot_df["gene_label"] = plot_df["gene"]

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, max(2.5, 0.4 * len(plot_df))))
        colors = ["#2F8F5B" if v > 0 else "#A8425C" for v in plot_df["shap_value"]]
        ax.barh(plot_df["gene_label"], plot_df["shap_value"], color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP contribution (log-odds, green = toward responder, red = toward non-responder)")
        ax.set_title(f"Top {len(plot_df)} genes driving the prediction for {sample_id}")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except ImportError:
        # Fallback if matplotlib isn't installed: use Streamlit's built-in bar chart
        st.bar_chart(plot_df.set_index("gene_label")["shap_value"])

    display_cols = explanation_df[["gene", "direction", "shap_value", "sample_value_log2", "training_median_log2"]].copy()
    display_cols.columns = ["Gene", "Direction", "SHAP contribution", "This sample (log2)", "Training median (log2)"]
    display_cols["SHAP contribution"] = display_cols["SHAP contribution"].round(3)
    display_cols["This sample (log2)"] = display_cols["This sample (log2)"].round(3)
    display_cols["Training median (log2)"] = display_cols["Training median (log2)"].round(3)
    st.dataframe(display_cols, use_container_width=True, hide_index=True)


if uploaded_file is not None:
    ok, remaining = check_rate_limit()
    if not ok:
        st.error(
            f"You've reached the limit of {MAX_REQUESTS_PER_WINDOW} uploads per hour for this "
            f"session. Please try again later."
        )
        st.stop()

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        st.error(f"File is {size_mb:.1f}MB, which exceeds the {MAX_FILE_SIZE_MB}MB limit.")
        st.stop()

    try:
        raw_df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read this file as CSV: {e}")
        st.stop()

    validation_errors = validate_dataframe(raw_df)
    if validation_errors:
        st.error("This file can't be processed:")
        for err in validation_errors:
            st.write(f"- {err}")
        st.stop()

    record_request()
    st.caption(f"{remaining - 1} uploads remaining this hour for this session.")
    st.write(f"Loaded **{len(raw_df)}** sample(s) with **{raw_df.shape[1] - 1}** expression columns.")

    try:
        results, prepared, present, missing, selected_features = run_pipeline(raw_df, already_log)
    except Exception as e:
        st.error(f"Something went wrong while processing this file: {e}")
        st.stop()

    n_expected = len(selected_features)
    if missing:
        st.warning(
            f"**{len(missing)} of {n_expected}** genes the model expects were not found in "
            f"your file and were imputed using training-set medians instead of your actual data. "
            f"Found {len(present)}/{n_expected}. "
            f"Missing genes (first 15 shown): {', '.join(missing[:15])}"
            + ("..." if len(missing) > 15 else "")
        )
        if len(present) < n_expected * 0.5:
            st.error(
                "Fewer than half the expected genes were found. Treat this prediction as "
                "very low confidence -- most of the model's input is coming from population "
                "medians, not your actual sample."
            )
    else:
        st.success(f"All {n_expected} expected genes were found in your file. No imputation needed.")

    st.subheader("Predictions")
    display_df = results.copy()
    display_df["responder_probability"] = display_df["responder_probability"].round(4)
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    if len(results) > 1:
        st.subheader("Responder probability by sample")
        chart_df = results.set_index("sample_id")[["responder_probability"]]
        st.bar_chart(chart_df, color="#146C6B")

        n_responders = (results["predicted_response"] == "responder").sum()
        st.caption(
            f"{n_responders} of {len(results)} sample(s) predicted responder "
            f"({n_responders / len(results):.0%})."
        )
    else:
        row = results.iloc[0]
        st.metric(
            label=str(row["sample_id"]),
            value=row["predicted_response"].replace("_", " ").title(),
            delta=f"{row['responder_probability']:.1%} responder probability",
            delta_color="off",
        )

    # --- SHAP explainability ---
    model, _, training_medians = get_model_artifacts()

    st.subheader("Why this prediction? (SHAP explanation)")
    st.caption(
        "For each sample, shows the genes that contributed most to pushing the prediction "
        "toward responder (green) or non-responder (red), compared to a typical training "
        "patient (the training-set median for each gene). This is exact for the deployed "
        "logistic regression model -- the baseline plus all contributions add up to the "
        "model's raw score."
    )
    try:
        explanations = explain_predictions(model, prepared, selected_features, training_medians, top_n=SHAP_TOP_N)
        if len(results) == 1:
            sample_id = results.iloc[0]["sample_id"]
            render_explanation(sample_id, explanations[sample_id])
        else:
            sample_choice = st.selectbox("Select a sample to explain", results["sample_id"].tolist())
            render_explanation(sample_choice, explanations[sample_choice])
    except TypeError as e:
        st.info(f"SHAP explanation isn't available for this model: {e}")
    except Exception as e:
        st.warning(f"Could not generate SHAP explanation: {e}")

    csv_buffer = io.StringIO()
    results.to_csv(csv_buffer, index=False)
    st.download_button(
        "Download predictions as CSV",
        data=csv_buffer.getvalue(),
        file_name="predictions.csv",
        mime="text/csv",
    )

    st.caption(
        "Reminder: this assumes your sample's expression values are on a comparable scale to "
        "the training cohorts (melanoma/bladder/NSCLC, see README). Samples from a very "
        "different sequencing platform or lab should be treated as exploratory."
    )
else:
    st.info("Upload a CSV to get started, or see the format instructions above.")

st.divider()
st.caption(
    "Model: logistic regression trained on merged, ComBat-corrected expression data from "
    "published immunotherapy cohorts. Not for clinical use."
)
