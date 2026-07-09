# ExAI-PanC — Explainable AI for Pancreatic Cancer

[![Python](https://img.shields.io/badge/Python-3.10%2B-2563eb?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-00AB6B?style=flat&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![DOI](https://img.shields.io/badge/DOI-10.1101%2F2024.01.01.24300100-3b82f6)](https://doi.org/10.1101/2024.01.01.24300100)

**ExAI-PanC** is an end-to-end AI-powered clinical decision support system for **Pancreatic Ductal Adenocarcinoma (PDAC)**. It integrates multi-omics machine learning, explainable AI (SHAP), and a **Clinical Digital Twin** — a dynamic 3D anatomical visualization — to provide risk assessment, cancer detection, staging, treatment recommendation, and survival prediction for individual patients.

> **Research Use Only.** This tool is a research prototype and is not approved for clinical decision-making.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Pipeline & Methodology](#pipeline--methodology)
- [Key Outcomes](#key-outcomes)
- [Screenshots](#screenshots)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Technical Stack](#technical-stack)
- [License](#license)
- [Author](#author)

---

## Overview

Pancreatic cancer has a **5-year survival rate of just 12%**, largely due to late detection and limited treatment options. ExAI-PanC addresses this by combining:

- **Multi-omics machine learning** trained on TCGA-PAAD (n=183) and GTEx (n=362) datasets
- **Variational Autoencoders (VAE)** for gene expression compression and latent feature discovery
- **XGBoost classifiers** for cancer detection and staging
- **Random Survival Forest** for survival analysis
- **SHAP (SHapley Additive exPlanations)** for model interpretability
- **Compartmental ODE model** for disease progression simulation
- **Clinical Digital Twin** — an interactive 3D torso visualization with organ-level risk mapping

The system processes patient clinical data (demographics, biomarkers, risk factors) and outputs a comprehensive clinical report with cancer probability, AJCC stage, treatment recommendations, survival projections, organ-level risk scores, and SHAP-based explanations.

---

## Features

| Feature | Description |
|---|---|
| **Cancer Detection** | XGBoost classifier with AUC-ROC **0.999** (5-fold CV on TCGA vs GTEx) |
| **Cancer Staging** | AJCC Stage I–IV prediction from clinical + biomarker data |
| **Survival Analysis** | Random Survival Forest with C-index **0.857**; 1-year and 2-year survival probabilities |
| **Treatment Recommendation** | Response probability for FOLFIRINOX, Gemcitabine + nab-paclitaxel, FOLFOX, Best Supportive Care |
| **SHAP Explainability** | Per-feature attribution scores identifying cancer-driving vs normal-pushing factors |
| **Clinical Digital Twin** | 3D interactive organ risk heatmap, disease progression simulation, treatment response visualization |
| **Biomarker Timeline** | 12-month CA 19-9, CEA, Bilirubin, Amylase trajectory projection |
| **Organ Risk Mapping** | 8-organ risk assessment (Pancreas, Liver, Bile Duct, Lymph Nodes, Duodenum, Stomach, Spleen, Lung) |
| **CSV Import/Export** | Batch patient data loading and clinical report download |
| **Compartmental ODE Model** | APS-inspired differential equation model for tumor-drug-immune dynamics |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                     ExAI-PanC System Architecture                      │
└──────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT LAYER                                   │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────────────────┐  │
│  │ Demographics │  │  Biomarkers  │  │  Risk Factors & Genetics  │  │
│  │ Age, Sex     │  │ CA19-9, CEA  │  │ Smoking, Diabetes, BRCA   │  │
│  └──────┬───────┘  │ Bilirubin,   │  │ Pancreatitis, Family Hx   │  │
│         │          │ Amylase, Wt  │  └──────────────┬────────────┘  │
│         │          └──────┬───────┘                 │               │
│         └─────────────────┼─────────────────────────┘               │
│                           ▼                                         │
│                   DataPipeline (preprocessing)                       │
└─────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                      PREDICTION ENSEMBLE                             │
│                                                                      │
│  ┌────────────────────┐  ┌────────────────┐  ┌───────────────────┐  │
│  │  VAE Encoder       │  │  XGBoost       │  │  Random Survival  │  │
│  │  (Gene Expression  │  │  Detection +   │  │  Forest +         │  │
│  │   → Latent Space)  │  │  Stage Models  │  │  Clinical Model   │  │
│  └────────────────────┘  └────────────────┘  └───────────────────┘  │
│                                                                      │
│  ┌────────────────────┐  ┌────────────────┐  ┌───────────────────┐  │
│  │  Treatment Model   │  │  SHAP Engine   │  │  Organ Risk       │  │
│  │  (Heuristic +      │  │  (XGBoost      │  │  Computation      │  │
│  │   Clinical Rules)  │  │  Contributions)│  │  (8 Organs)       │  │
│  └────────────────────┘  └────────────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                        DIGITAL TWIN                                  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    DigitalTwinEngine                           │   │
│  │  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌────────────┐  │   │
│  │  │ Anatomy  │  │ Disease   │  │ Compart  │  │ Simulation │  │   │
│  │  │ Loader   │  │ Progress  │  │ -mental  │  │ Engine     │  │   │
│  │  │ (Mesh3D) │  │ -ion      │  │ ODE      │  │ + What-If  │  │   │
│  │  └──────────┘  └───────────┘  └──────────┘  └────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Visualization Layer (Plotly)                                  │   │
│  │  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌───────────────┐  │   │
│  │  │ 3D Organ │  │ Gauge    │  │ SHAP   │  │ KM Survival   │  │   │
│  │  │ Heatmap  │  │ Chart    │  │ Bar    │  │ Curve         │  │   │
│  │  └──────────┘  └──────────┘  └────────┘  └───────────────┘  │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────────┐
│                      OUTPUT LAYER                                    │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ Clinical │  │ Organ    │  │ SHAP     │  │ Digital Twin      │  │
│  │ Report   │  │ Risk     │  │ Explana  │  │ Interactive       │  │
│  │ (Text)   │  │ Summary  │  │ -tions   │  │ Visualization     │  │
│  └──────────┘  └──────────┘  └──────────┘  └───────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Module Breakdown

| Module | Path | Description |
|---|---|---|
| **Prediction Engine** | `prediction_engine.py` | Core prediction pipeline: cancer probability, staging, survival, treatment, SHAP, organ risks |
| **Digital Twin Core** | `digital_twin/core/` | `DigitalTwinEngine`, `TwinState`, `SimulationEngine`, `RecommendationEngine` |
| **Anatomy** | `digital_twin/anatomy/` | 3D organ mesh models, rendering, lesion mapping, mesh utilities |
| **Prediction** | `digital_twin/prediction/` | `DetectionPredictor`, `StagePredictor`, `SurvivalPredictor`, `PredictionEnsemble` |
| **Simulation** | `digital_twin/simulation/` | Compartmental ODE twin, disease progression, intervention simulation |
| **Explainability** | `digital_twin/explainability/` | SHAP engine, biomarker mapping, clinical overlay HTML generators |
| **Visualization** | `digital_twin/visualization/` | Plotly chart generators: gauge, SHAP, treatment, KM, organ radar, biomarker timeline |
| **Training Pipeline** | `train_models.py` | End-to-end training on TCGA + GTEx data: VAE, XGBoost, Random Survival Forest |
| **Dashboard** | `app.py` / `panc_dashboard.py` | Streamlit web application (refactored and original versions) |

---

## Pipeline & Methodology

### 1. Data Acquisition & Preprocessing
- **TCGA-PAAD**: 183 pancreatic adenocarcinoma patients with RNA-seq, clinical, and mutation data
- **GTEx Pancreas**: 362 healthy pancreas tissue samples
- **Feature Selection**: Top 500 differentially expressed genes by fold-change and FDR
- **Clinical Features**: Age, sex, smoking, alcohol, diabetes, family history, CA 19-9, CEA, bilirubin, amylase, weight loss

### 2. Variational Autoencoder (VAE)
- **Architecture**: 500 → 256 → 128 → 64 (latent) → 128 → 256 → 500
- **Purpose**: Compress high-dimensional gene expression into a 64-dimensional latent space, capturing biological variability
- **Output**: Feature embeddings used alongside clinical data for downstream tasks

### 3. Detection Model (XGBoost)
- **Input**: VAE latent features (64-dim) + clinical features
- **Output**: Cancer probability (0–100%)
- **Performance**: AUC-ROC **0.999**, Accuracy **99.3%** (5-fold cross-validation)

### 4. Staging Model
- **Heuristic-driven** probability-based AJCC stage assignment using cancer probability + CA 19-9 + bilirubin thresholds
- Stages: Non-malignant → Stage I → Stage II → Stage III → Stage IV

### 5. Survival Model (Random Survival Forest)
- **Input**: Age, sex, cancer probability, stage
- **Output**: Risk score, median survival (months), 1-year / 2-year survival probabilities
- **Performance**: C-index **0.857**

### 6. Treatment Response Model
- Probabilistic response estimation for four regimens:
  - **FOLFIRINOX** (preferred for younger, fit patients)
  - **Gemcitabine + nab-paclitaxel**
  - **FOLFOX**
  - **Best Supportive Care**

### 7. SHAP Explainability
- **TreeSHAP** from XGBoost for age and sex contributions
- **Heuristic SHAP** for biomarkers and risk factors (CA 19-9, bilirubin, weight loss, CEA, diabetes, smoking, family history, pancreatitis)
- Per-feature attribution visualized as horizontal bar plot

### 8. Compartmental ODE Model (APS-Inspired)
- **6 compartments**: Tumor (T), CA 19-9 (CA), CEA (CE), Bilirubin (B), Drug (D), Immune (I)
- **Gompertzian tumor growth** with drug effect and immune response coupling
- **Intervention simulation**: Apply treatment at arbitrary timepoints and observe projected trajectories

### 9. Clinical Digital Twin
- **11 organs**: Pancreas, Liver, Gallbladder, Bile Duct, Duodenum, Stomach, Spleen, Lungs, Lymph Nodes, Heart, Kidneys
- **Render modes**: Risk Heatmap, Disease Progression, Treatment Response, Compartmental ODE
- **3D anatomical visualization** using Plotly Mesh3d with color-coded risk levels (green → yellow → orange → red)

---

## Key Outcomes

| Metric | Value | Model |
|---|---|---|
| **AUC-ROC** | **0.999** | XGBoost Detection (5-fold CV) |
| **Accuracy** | **99.3%** | XGBoost Detection |
| **C-index** | **0.857** | Random Survival Forest |
| **Training Cohort** | **183 TCGA + 362 GTEx** | 545 total samples |
| **VAE Latent Dim** | **64** | Compressed from 500 genes |
| **Organs Modeled** | **11** | Interactive 3D visualization |
| **Treatment Options** | **4** | Evidence-based chemotherapy regimens |

---

## Screenshots

> *Note: Run the Streamlit app locally to generate live interactive screenshots of each view below.*

### Home Dashboard
```
┌─────────────────────────────────────────────────────────────────────┐
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ 0.999    │  │ 99.3%    │  │ 0.857    │  │ 138 Patients    │  │
│  │ AUC-ROC  │  │ Accuracy │  │ C-index  │  │ Training Cohort  │  │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘  │
│                                                                      │
│  ExAI-PanC — Pancreatic Cancer Decision Support                      │
│  Risk Assessment · Staging · Treatment Guidance                      │
│                                                                      │
│  [➡ Start — Enter Patient Data]                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Patient Data Entry
*Demographic inputs (age, sex), biomarker values (CA 19-9, CEA, bilirubin, amylase, weight loss), risk factors (smoking, alcohol, diabetes, family history, pancreatitis, genetic variants), CSV upload support.*

### Detection — Cancer Probability Gauge
*A color-coded gauge chart (green → yellow → red) displaying the patient's cancer probability score with confidence level indicator.*

### Stage & Treatment
*AJCC stage prediction card with stage indicator, treatment response bar chart ranking FOLFIRINOX, Gemcitabine+nab-paclitaxel, FOLFOX, and Best Supportive Care by predicted efficacy.*

### Survival Analysis — Kaplan-Meier Curve
*Risk-adjusted survival curve comparing the patient against low, medium, and high-risk reference populations with median survival annotation.*

### SHAP Explainability
*Horizontal bar chart showing per-feature SHAP values — red bars push toward cancer, blue bars push toward normal. Top attributions listed with magnitude bars.*

### Clinical Digital Twin — 3D Organ Risk Heatmap
*Interactive 3D torso visualization with 11 color-coded organs. Hover for organ-level risk scores, health scores, and biomarker breakdown. Modes: Risk Heatmap, Disease Progression (0–36 months), Treatment Response, Compartmental ODE.*

### Clinical Report
*Comprehensive downloadable report (TXT + CSV) summarizing all predictions, organ risks, and recommended treatment.*

---

## Installation

### Prerequisites
- Python **3.10+**
- pip / conda

### Clone & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/XAI-PanC.git
cd XAI-PanC

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Download Pre-trained Models

The `models/` directory contains pre-trained model artifacts:

| File | Description |
|---|---|
| `vae_encoder.pt` | PyTorch VAE encoder (500→64 latent dim) |
| `xgb_detection.json` | XGBoost cancer detection classifier |
| `xgb_clinical_detection.json` | XGBoost clinical detection model |
| `xgb_clinical_stage.json` | XGBoost clinical stage model |
| `rf_clinical_survival.pkl` | Random Forest clinical survival model |
| `rf_survival.pkl` | Random Survival Forest (full features) |
| `scaler.pkl` | Feature StandardScaler |
| `clinical_medians.pkl` | Clinical feature medians for imputation |
| `sample_input.pkl` | Template input structure |
| `metadata.json` | Model metadata, top genes, detection metrics |

---

## Usage

### Launch the Dashboard

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

### Workflow

1. **Patient Data** → Enter patient demographics, biomarkers, and risk factors (or upload CSV)
2. **Run Prediction** → Click "Run Prediction" to execute the full analysis pipeline
3. **Detection** → View cancer probability gauge and biomarker status
4. **Stage & Treatment** → Review AJCC stage and treatment response predictions
5. **Survival** → Examine Kaplan-Meier survival curve and risk metrics
6. **ExAI — SHAP** → Explore per-feature attribution explanations
7. **Clinical Digital Twin** → Interact with 3D organ risk heatmap and disease progression simulation
8. **Report** → Download comprehensive clinical report (TXT) and organ risks (CSV)

### Training Models (Optional)

```bash
python train_models.py
```

Requires PAN-DATASET/ directory with TCGA-PAAD and GTEx data (not included; see data sources below).

### Run as Module

```python
from prediction_engine import run_full_prediction

inputs = {
    'age': 65, 'sex': 'Male',
    'ca199': 450.0, 'cea': 5.5,
    'bilirubin': 2.1, 'amylase': 180,
    'weight_loss_pct': 8, 'smoking': 'Former',
    'alcohol': 'Moderate (<14/week)',
    'diabetes': 'New-onset (<2 years)',
    'family_history': True, 'pancreatitis': False,
    'genetic': False,
}

result = run_full_prediction(inputs)
print(f"Cancer Probability: {result['prob']*100:.1f}%")
print(f"Stage: {result['stage_label']}")
print(f"Recommended Treatment: {result['best_tx']}")
```

---

## Project Structure

```
XAI-PanC/
├── app.py                                    # Streamlit dashboard (refactored)
├── panc_dashboard.py                         # Streamlit dashboard (original)
├── prediction_engine.py                      # Core prediction pipeline
├── train_models.py                           # Model training (TCGA + GTEx)
├── requirements.txt                          # Python dependencies
│
├── digital_twin/                             # Clinical Digital Twin package
│   ├── __init__.py                           # Package exports
│   ├── ai_twin_engine.py                     # Legacy twin engine
│   ├── anatomy_loader.py                     # 3D mesh loading
│   ├── biomarker_mapper.py                   # Organ-biomarker associations
│   ├── clinical_overlay.py                   # HTML overlay generators
│   ├── compartmental_twin.py                 # APS ODE model
│   ├── disease_progression.py                # Progression simulation
│   ├── lesion_mapper.py                      # Lesion mapping
│   ├── mesh_utils.py                         # Mesh manipulation utilities
│   ├── organ_animation.py                    # Organ animation
│   ├── organ_models.py                       # Organ/Lesion data classes
│   ├── organ_renderer.py                     # Plotly Mesh3D rendering
│   │
│   ├── core/                                 # Core engine modules
│   │   ├── twin_engine.py                    # DigitalTwinEngine
│   │   ├── twin_state.py                     # TwinState container
│   │   ├── simulation_engine.py              # Simulation orchestrator
│   │   └── recommendation_engine.py          # Treatment recommendations
│   │
│   ├── pipeline/                             # Data pipeline
│   │   ├── patient_profile.py                # PatientProfile model
│   │   └── data_pipeline.py                  # DataPipeline (CSV, preprocessing)
│   │
│   ├── prediction/                           # Prediction models
│   │   ├── ensemble.py                       # PredictionEnsemble
│   │   ├── detection_model.py                # DetectionPredictor
│   │   ├── stage_model.py                    # StagePredictor
│   │   └── survival_model.py                 # SurvivalPredictor
│   │
│   ├── simulation/                           # Simulation models
│   │   ├── compartmental_twin.py             # PancreaticTwinModel (ODE)
│   │   ├── disease_progression.py            # Disease progression
│   │   └── intervention_simulator.py         # What-if analysis
│   │
│   ├── explainability/                       # Explainability
│   │   ├── shap_engine.py                    # SHAP computation
│   │   ├── biomarker_mapper.py               # Organ-biomarker mapping
│   │   └── clinical_overlay.py               # HTML overlays
│   │
│   ├── visualization/                        # Visualization
│   │   ├── charts.py                         # Gauge, SHAP, KM, treatment charts
│   │   └── organ_animation.py                # Animated organ views
│   │
│   └── config/                               # Configuration
│       └── config.py                         # Paths, constants, organ definitions
│
├── models/                                   # Pre-trained model artifacts
│   ├── vae_encoder.pt                        # VAE weights
│   ├── xgb_detection.json                    # XGBoost detection
│   ├── xgb_clinical_detection.json           # XGBoost clinical detection
│   ├── xgb_clinical_stage.json               # XGBoost clinical stage
│   ├── rf_clinical_survival.pkl              # RF clinical survival
│   ├── rf_survival.pkl                       # RF survival (full)
│   ├── scaler.pkl                            # Feature scaler
│   ├── clinical_medians.pkl                  # Clinical medians
│   ├── sample_input.pkl                      # Sample input template
│   └── metadata.json                         # Top genes & model metadata
│
├── 02_data_preprocessing.ipynb               # Data preprocessing notebook
├── 03_VAE_BiologicalNetwork_GAT.ipynb        # VAE training notebook
├── 04_Clinical_Fusion_XGBoost_DeepSurv.ipynb # XGBoost + DeepSurv notebook
├── 05_SHAP_Explainability_Dashboard.ipynb    # SHAP analysis notebook
├── 06_Model_Evaluation_Report.ipynb          # Model evaluation notebook
├── 08_Model1_Verification_Tests_FIXED.ipynb  # Verification tests notebook
│
├── .gitignore
└── README.md
```

---

## Technical Stack

| Layer | Technology |
|---|---|
| **Frontend** | Streamlit, Plotly, HTML/CSS |
| **Machine Learning** | XGBoost, scikit-learn, scikit-survival |
| **Deep Learning** | PyTorch (VAE) |
| **Explainability** | TreeSHAP, heuristic feature attribution |
| **Survival Analysis** | Random Survival Forest |
| **Simulation** | Numerical ODE integration (compartmental model) |
| **Data Processing** | NumPy, Pandas, scikit-learn |
| **Visualization** | Plotly (Surface, Mesh3d, Bar, Scatter), SVG |
| **Notebooks** | Jupyter (research & development) |

---

## Data Sources

- **TCGA-PAAD**: The Cancer Genome Atlas Pancreatic Adenocarcinoma dataset ([NCI GDC](https://portal.gdc.cancer.gov/projects/TCGA-PAAD))
- **GTEx**: Genotype-Tissue Expression pancreas samples ([GTEx Portal](https://gtexportal.org/))
- **Note**: Raw data files are not included in this repository due to size and licensing restrictions.

---

## License

This project is licensed under the **MIT License**. See `LICENSE` for details.

---

## Author

**Athiyaman M**

[![GitHub](https://img.shields.io/badge/GitHub-AthiyamanM-181717?style=flat&logo=github&logoColor=white)](https://github.com/AthiyamanM)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-AthiyamanM-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/AthiyamanM)

*Built with passion for AI-powered healthcare — combining multi-omics machine learning, explainable AI, and digital twin technology to improve pancreatic cancer outcomes.*
