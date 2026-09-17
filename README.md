# AI-Based Personalized Immunotherapy Response Prediction

Pan-cancer, explainable multi-omics pipeline that predicts whether a patient will
respond to immune checkpoint inhibitor (ICI) therapy, and asks a specific,
publishable research question:

> **Do SHAP-derived response thresholds (TMB, immune-cell fractions, checkpoint
> gene expression, etc.) generalize across cancer types, or are they
> cancer-type-specific?**

This design deliberately builds on and extends three published approaches:

| Paper | What it contributes | What it doesn't do |
|---|---|---|
| Tran/Waddell et al. 2026 (*Sci Rep*) | Multi-omic (clinical+DNA+RNA) ML + SHAP-derived numeric thresholds, rigorous leave-one-cohort-out validation | Stays within melanoma only |
| NetBio, Park et al. 2022 (*Nat Commun*) | Network-propagated (pathway-level) features that generalize across 3 cancer types | No explainable thresholds |
| EaSIeR / multi-omics review (*npj Digital Medicine*) | Pretrains on TCGA immune-response scores without needing outcome labels, transfers across 4 cancer types | Doesn't combine with threshold discovery |

This repo's pipeline = network-propagated features (NetBio) + label-free TCGA
pretraining (EaSIeR) + SHAP threshold discovery and **cross-cancer-type
transfer testing** (extends Tran/Waddell) — the specific combination none of
the three papers did.

---

## 0. Prerequisites

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` covers the Python side. Two external/web tools are used
outside Python and are **not pip-installable**:

- **CIBERSORTx** (immune cell deconvolution) — free academic account required
  at https://cibersortx.stanford.edu. You upload a gene expression matrix and
  download cell-fraction results; there's no public API, so this step is
  manual (see Phase 2).
- **STAR / Salmon** (if you start from raw FASTQ instead of pre-computed
  expression matrices) — only needed if your chosen cohorts don't already
  provide processed TPM/count matrices. Most public ICI cohorts (below) do.

---

## 1. Project layout

```
immunotherapy-response-prediction/
├── README.md                     <- this file
├── requirements.txt
├── data/
│   ├── raw/                      <- downloaded cohort files go here
│   └── processed/                <- cleaned, merged feature tables
├── src/
│   ├── data_loading.py           <- Phase 1: load/harmonize cohorts
│   ├── preprocessing.py          <- Phase 1: QC, normalization, batch correction
│   ├── network_propagation.py    <- Phase 2: NetBio-style pathway features
│   ├── feature_engineering.py    <- Phase 2: TMB, checkpoint genes, immune scores
│   ├── tcga_pretraining.py       <- Phase 3: EaSIeR-style label-free pretraining
│   ├── feature_selection.py      <- Phase 4: MWU screen + SHAP collinearity pruning
│   ├── models.py                 <- Phase 5: train/evaluate LR, RF, SVM, ensembles
│   ├── cross_cohort_validation.py<- Phase 6: leave-one-cohort-out, cross-cancer
│   ├── shap_analysis.py          <- Phase 7: SHAP importance + threshold discovery
│   └── thresholds.py             <- Phase 7: cross-cancer threshold comparison
├── main.py                       <- runs the whole pipeline end-to-end
└── outputs/                      <- metrics, plots, SHAP tables land here
```

---

## 2. Step-by-step workflow

### Phase 1 — Get data and harmonize it

**Where to get labeled ICI cohorts** (RNA-seq/DNA + documented CR/PR/SD/PD):

| Cancer type | Cohort | Access |
|---|---|---|
| Melanoma | Hugo et al. 2016 | ENA `PRJNA312948` |
| Melanoma | Riaz et al. 2017 | ENA `PRJNA359359` / `PRJNA356761` |
| Melanoma | Van Allen et al. 2015 | dbGaP `phs000452` |
| Melanoma | Liu et al. 2019 | dbGaP `phs001041` |
| Melanoma | Gide et al. 2019 | ENA `PRJEB23709` |
| Bladder/urothelial | IMvigor210 (Mariathasan et al. 2018) | R package — see `scripts/export_imvigor210_easierdata.R` (recommended) or `scripts/export_imvigor210_corebiologies.R` (full cohort). Not a standalone GEO series. |
| Bladder/urothelial (independent validation cohort, NOT the same as IMvigor210) | GSE176307 | GEO |
| Gastric | multiple GEO cohorts | GEO, search "gastric cancer anti-PD-1 RNA-seq" |
| Renal | Braun et al. 2020 (CheckMate 025/214) | cBioPortal / supplementary data |
| Pan-cancer background (unlabeled) | TCGA | `gdc-client` via GDC Data Portal (`portal.gdc.cancer.gov`) |

Start with **2–3 cancer types** (e.g., melanoma + bladder) — that's enough to
test cross-cancer transfer without an unmanageable data-wrangling job.

```bash
python -m src.data_loading --cohort hugo2016 --cancer melanoma --out data/raw/
python -m src.data_loading --cohort imvigor210 --cancer bladder --out data/raw/
```

Then clean/normalize and merge into one table:

```bash
python -m src.preprocessing --input data/raw/ --output data/processed/merged_expression.csv
```

### Phase 2 — Feature engineering

1. Run CIBERSORTx (manual, web-based) on `merged_expression.csv` → download
   `CIBERSORTx_Results.csv` into `data/raw/`.
2. Compute checkpoint gene expression, TMB (if mutation data available), and
   immune gene signatures:

```bash
python -m src.feature_engineering \
    --expression data/processed/merged_expression.csv \
    --cibersort data/raw/CIBERSORTx_Results.csv \
    --output data/processed/features.csv
```

3. Build network-propagated (pathway-level) features:

```bash
python -m src.network_propagation \
    --expression data/processed/merged_expression.csv \
    --output data/processed/network_features.csv
```

### Phase 3 (optional, stretch) — TCGA label-free pretraining

```bash
python -m src.tcga_pretraining --tcga-dir data/raw/tcga/ --output data/processed/tcga_pretrained_scores.csv
```

### Phase 4 — Feature selection

```bash
python -m src.feature_selection \
    --features data/processed/features.csv \
    --network-features data/processed/network_features.csv \
    --labels data/processed/labels.csv \
    --output data/processed/selected_features.csv
```

### Phase 5 — Train and evaluate models

```bash
python -m src.models \
    --data data/processed/selected_features.csv \
    --output outputs/model_results.json
```

### Phase 6 — Cross-cohort / cross-cancer-type validation

```bash
python -m src.cross_cohort_validation \
    --data data/processed/selected_features.csv \
    --output outputs/cross_validation_results.csv
```

### Phase 7 — SHAP explainability + threshold discovery

```bash
python -m src.shap_analysis \
    --data data/processed/selected_features.csv \
    --output outputs/shap_results/

python -m src.thresholds \
    --shap-dir outputs/shap_results/ \
    --output outputs/threshold_comparison.csv
```

`threshold_comparison.csv` is the core novel-contribution output: it shows,
per feature, whether the SHAP-derived cutoff differs by cancer type.

### Run everything at once

```bash
python main.py --cancers melanoma bladder --config config.yaml
```

---

## 3. Evaluation checklist

- [ ] Report AUC-ROC, weighted F1, precision, recall, confusion matrix — never accuracy alone.
- [ ] Use repeated stratified k-fold (imbalanced classes are the norm here).
- [ ] Always validate on a held-out cohort the model never saw in training.
- [ ] Report calibration (Brier score) on the cross-cancer test fold specifically.
- [ ] Document where the model fails (e.g. stable disease, rare subtypes) — don't just report best-case AUC.

## 4. Known limitations to state explicitly in your write-up

- Public labeled cohorts are small (tens–low hundreds per cancer type); results are exploratory, not clinically validated.
- CIBERSORTx must be run separately per dataset/batch per its own usage guidance — do not pool raw counts across cohorts before deconvolution.
- SHAP thresholds are associational, not causal, and need prospective/clinical validation before any clinical use.
