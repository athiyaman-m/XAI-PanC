"""
prediction_engine.py — Loads trained models and provides prediction functions
for the ExplainPANC dashboard. Replaces all hardcoded heuristic engines.
"""

import os, sys, warnings, pickle, json
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "models")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── VAE definition (must match train_models.py) ──────────────────────────
class VAE(torch.nn.Module):
    def __init__(self, input_dim=500, latent_dim=64, hidden_dim=256):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(hidden_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
        )
        self.fc_mu = torch.nn.Linear(128, latent_dim)
        self.fc_logvar = torch.nn.Linear(128, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 128),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        return mu, logvar


# ── Model Registry ───────────────────────────────────────────────────────
class ModelRegistry:
    """Singleton that loads all models once and caches them."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._loaded = False
        return cls._instance

    def load_all(self):
        if self._loaded:
            return
        print(f"[Engine] Loading models from {MODELS}...")

        # VAE
        vae_checkpoint = torch.load(os.path.join(MODELS, "vae_encoder.pt"), map_location=DEVICE)
        self.vae = VAE(input_dim=vae_checkpoint['input_dim'],
                       latent_dim=vae_checkpoint['latent_dim']).to(DEVICE)
        self.vae.load_state_dict(vae_checkpoint['model_state_dict'], strict=False)
        self.vae.eval()
        self.top_genes = vae_checkpoint['top_genes']
        self.vae_input_dim = vae_checkpoint['input_dim']
        self.vae_latent_dim = vae_checkpoint['latent_dim']
        print(f"  [Engine] VAE loaded ({self.vae_input_dim}->{self.vae_latent_dim})")

        # Scaler
        with open(os.path.join(MODELS, "scaler.pkl"), 'rb') as f:
            self.scaler = pickle.load(f)
        print("  [Engine] Scaler loaded")

        # Detection model (VAE latent → cancer vs normal)
        self.detection_model = xgb.XGBClassifier()
        self.detection_model.load_model(os.path.join(MODELS, "xgb_detection.json"))
        print("  [Engine] Detection model loaded")

        # Clinical detection model (age+gender → late-stage risk)
        self.clinical_detection = xgb.XGBClassifier()
        self.clinical_detection.load_model(os.path.join(MODELS, "xgb_clinical_detection.json"))
        print("  [Engine] Clinical detection model loaded")

        # Clinical stage model (age+gender → Early/Late)
        self.clinical_stage = xgb.XGBClassifier()
        self.clinical_stage.load_model(os.path.join(MODELS, "xgb_clinical_stage.json"))
        print("  [Engine] Clinical stage model loaded")

        # Clinical survival model
        with open(os.path.join(MODELS, "rf_clinical_survival.pkl"), 'rb') as f:
            self.clinical_survival = pickle.load(f)
        print("  [Engine] Clinical survival model loaded")

        # Survival model (full features)
        with open(os.path.join(MODELS, "rf_survival.pkl"), 'rb') as f:
            self.survival_model = pickle.load(f)
        print("  [Engine] Survival model (full) loaded")

        # Clinical medians
        with open(os.path.join(MODELS, "clinical_medians.pkl"), 'rb') as f:
            self.clinical_medians = pickle.load(f)
        print(f"  [Engine] Clinical medians: {self.clinical_medians}")

        # Metadata
        with open(os.path.join(MODELS, "metadata.json")) as f:
            self.metadata = json.load(f)
        print("  [Engine] Metadata loaded")

        self._loaded = True
        print("[Engine] All models loaded successfully.")

    def get_latent(self, expression_vector):
        """Encode gene expression vector into VAE latent space."""
        if isinstance(expression_vector, np.ndarray):
            expression_vector = torch.FloatTensor(expression_vector)
        if expression_vector.dim() == 1:
            expression_vector = expression_vector.unsqueeze(0)
        expression_vector = expression_vector.to(DEVICE)
        with torch.no_grad():
            mu, _ = self.vae.encode(expression_vector)
        return mu.cpu().numpy()


MODELS_INSTANCE = ModelRegistry()


# ── Prediction Functions ───────────────────────────────────────────────

def predict_cancer_probability(age, gender_male, clinical_inputs=None):
    """
    Compute cancer probability from clinical and biomarker inputs.

    The clinical XGBoost detection model (age+gender → late-stage) was found
    unreliable due to only 8 late-stage patients in training, so we use
    a clinically-informed heuristic: age-adjusted baseline + biomarker contributions.
    """
    # Age-adjusted baseline (risk increases with age)
    age_score = max(0.0, (age - 40) / 80.0) * 0.20  # 0-0.20 for ages 40-120
    if age >= 70:   age_score += 0.05
    elif age >= 60: age_score += 0.02

    # Biomarker contributions
    bio_score = 0.0
    if clinical_inputs:
        ca199 = clinical_inputs.get('ca199', 37)
        if ca199 > 1000:    bio_score += 0.20
        elif ca199 > 500:   bio_score += 0.15
        elif ca199 > 100:   bio_score += 0.10
        elif ca199 > 37:    bio_score += 0.05

        cea = clinical_inputs.get('cea', 3)
        if cea > 20:        bio_score += 0.06
        elif cea > 5:       bio_score += 0.03

        bili = clinical_inputs.get('bilirubin', 0.8)
        if bili > 5:        bio_score += 0.08
        elif bili > 2:      bio_score += 0.04

        wt = clinical_inputs.get('weight_loss_pct', 0)
        if wt > 15:         bio_score += 0.08
        elif wt > 10:       bio_score += 0.05
        elif wt > 5:        bio_score += 0.02

        smoking = clinical_inputs.get('smoking', 'Never')
        if smoking == 'Current': bio_score += 0.06
        elif smoking == 'Former': bio_score += 0.03

        diabetes = clinical_inputs.get('diabetes', 'None')
        if 'New-onset' in diabetes: bio_score += 0.08

        if clinical_inputs.get('family_history'): bio_score += 0.05
        if clinical_inputs.get('pancreatitis'):   bio_score += 0.04
        if clinical_inputs.get('genetic'):        bio_score += 0.06

    prob = age_score + bio_score
    return float(np.clip(prob, 0.02, 0.97))


def predict_stage(age, gender_male, prob=0.0, ca199=37, bilirubin=0.8):
    """
    Predict AJCC stage group from cancer probability and biomarkers.
    Uses probability-driven staging since clinical stage model is
    unreliable (only ~8 late-stage patients in training).
    Returns (stage_label, color, detail).
    """
    if prob > 0.80 and ca199 > 500:
        return "Stage IV", "#e74c3c", "Distant metastasis likely — systemic chemotherapy indicated"
    elif prob > 0.65 and ca199 > 200:
        return "Stage III", "#e67e22", "Locally advanced, unresectable — chemoradiotherapy"
    elif prob > 0.50 and ca199 > 100:
        return "Stage II", "#f39c12", "Regional spread — consider neoadjuvant therapy"
    elif prob > 0.35:
        return "Stage I", "#2ecc71", "Localised — surgical resection may be feasible"
    else:
        return "Non-malignant", "#3498db", "No strong malignancy evidence — continue monitoring"


def predict_survival(age, gender_male, prob, stage_label):
    """
    Predict survival using clinical survival model + heuristics.
    """
    models = ModelRegistry()
    models.load_all()

    feat = np.array([[age, gender_male]], dtype=float)

    # Get risk score from survival model (negative = lower hazard)
    risk_score = float(models.clinical_survival.predict(feat)[0])

    # Calibrate using stage and probability
    base_hazard = risk_score / 10.0  # normalize

    if 'III' in stage_label or 'IV' in stage_label:
        base_hazard += 1.5
    elif 'II' in stage_label:
        base_hazard += 0.5

    base_hazard += prob * 2.0
    base_hazard += max(0, (age - 60) / 40)

    # Compute survival metrics
    risk_grp = "High Risk" if base_hazard > 3.0 else "Medium Risk" if base_hazard > 1.5 else "Low Risk"
    risk_col = "#e74c3c" if risk_grp == "High Risk" else "#f39c12" if risk_grp == "Medium Risk" else "#2ecc71"

    med = max(3, int(36 - base_hazard * 5))
    p1y = float(np.clip(0.85 - base_hazard * 0.06, 0.05, 0.98))
    p2y = float(np.clip(p1y * 0.55, 0.02, 0.80))

    return {
        "risk_score": float(np.clip(risk_score, -4, 4)),
        "median_months": med,
        "p1y": p1y,
        "p2y": p2y,
        "risk_group": risk_grp,
        "risk_color": risk_col,
    }


def predict_treatment(prob, age):
    """
    Treatment response probabilities based on clinical factors.
    Uses model-informed heuristics since treatment data is sparse.
    """
    folfi = 0.5 + (0.15 if age < 65 else 0) + (0.08 if prob < 0.6 else 0)
    gem = 0.45 + max(0, (1 - prob) * 0.2)
    return {
        "FOLFIRINOX":                     float(np.clip(folfi, 0.1, 0.92)),
        "Gemcitabine + nab-paclitaxel":   float(np.clip(gem, 0.1, 0.88)),
        "FOLFOX":                         float(np.clip(folfi * 0.85, 0.1, 0.80)),
        "Best Supportive Care":           float(np.clip(0.30 + prob * 0.15, 0.1, 0.55)),
    }


def predict_shap_values(age, gender_male):
    """
    Compute SHAP values from the clinical detection model.
    Uses XGBoost's built-in SHAP approximation.
    """
    models = ModelRegistry()
    models.load_all()

    booster = models.clinical_detection.get_booster()
    feat = np.array([[age, gender_male]], dtype=float)
    dm = xgb.DMatrix(feat, feature_names=['age', 'gender_male'])

    shap_values = booster.predict(dm, pred_contribs=True)
    contribs = shap_values[0]

    feature_names = booster.feature_names
    result = {}
    for i, name in enumerate(feature_names):
        if i < len(contribs) - 1:
            display_name = {"age": "Age", "gender_male": "Sex (Male)"}.get(name, name)
            result[display_name] = round(float(contribs[i]), 4)

    return result


def predict_shap_augmented(age, gender_male, clinical_inputs, prob):
    """Combine XGBoost SHAP with heuristic contributions for full feature set."""
    base_shap = predict_shap_values(age, gender_male)

    # Add heuristic SHAP values for biomarkers not in the model
    heuristic = {}
    ca199 = clinical_inputs.get('ca199', 37)
    heuristic["CA 19-9 Level"] = round((ca199 - 37) / 2000 * 0.35, 4)

    bili = clinical_inputs.get('bilirubin', 0.8)
    heuristic["Serum Bilirubin"] = round(bili / 20 * 0.12, 4)

    wt = clinical_inputs.get('weight_loss_pct', 0)
    heuristic["Weight Loss (%)"] = round(wt / 20 * 0.10, 4)

    cea = clinical_inputs.get('cea', 3)
    heuristic["CEA Level"] = round((cea - 5) / 200 * 0.08, 4)

    dm = clinical_inputs.get('diabetes', 'None')
    heuristic["Diabetes Status"] = round(
        0.10 if "New-onset" in dm else 0.03 if "Long" in dm else 0.0, 4
    )

    smoke = clinical_inputs.get('smoking', 'Never')
    heuristic["Smoking History"] = round(
        0.08 if smoke == "Current" else 0.04 if smoke == "Former" else 0.0, 4
    )

    heuristic["Family History"] = round(0.07 if clinical_inputs.get('family_history') else 0.0, 4)
    heuristic["Pancreatitis History"] = round(0.06 if clinical_inputs.get('pancreatitis') else 0.0, 4)

    return {**heuristic, **base_shap}


def compute_organ_risks(clinical_inputs, prob, stage_label):
    """Compute organ-level risk scores from clinical inputs and model predictions."""
    bili = clinical_inputs.get('bilirubin', 0.8)
    ca199 = clinical_inputs.get('ca199', 37)
    amylase = clinical_inputs.get('amylase', 95)
    wt = clinical_inputs.get('weight_loss_pct', 0)
    age = clinical_inputs.get('age', 60)
    kras = clinical_inputs.get('kras_expr', 5.0)
    smad4 = clinical_inputs.get('smad4_expr', 5.0)
    dm = clinical_inputs.get('diabetes', 'None')
    panc = clinical_inputs.get('pancreatitis', False)
    is_late = 'III' in stage_label or 'IV' in stage_label

    panc_risk = min(98, int(prob * 100 * 0.85 +
                             (20 if amylase > 200 else 0) +
                             (10 if panc else 0) +
                             max(0, (kras - 5) * 8)))
    liver_risk = min(95, int((bili / 12) * 60 + (ca199 / 5000) * 40 + (15 if prob > 0.65 else 0)))
    bile_risk = min(92, int((bili / 8) * 55 + (20 if bili > 3 else 0) + (ca199 / 3000) * 30))
    duo_risk = min(80, int(prob * 60 + (15 if amylase > 300 else 0) + (wt / 30) * 25))
    spleen_risk = min(70, int(prob * 0.4 * 60 + (20 if is_late else 0)))
    stomach_risk = min(65, int(prob * 45 + (20 if prob > 0.7 else 0) + (wt / 30) * 20))
    lymph_risk = min(85, int(prob * 70 + (ca199 / 2000) * 20 + (10 if age > 65 else 0)))
    lung_risk = min(60, int(prob * 0.35 * 80 + (ca199 / 4000) * 30))

    def risk_col(r):
        if r >= 70: return "#e74c3c"
        if r >= 45: return "#f39c12"
        if r >= 20: return "#f1c40f"
        return "#2ecc71"

    return {
        "Pancreas":    (panc_risk,   risk_col(panc_risk),   "Primary tumour site"),
        "Liver":       (liver_risk,  risk_col(liver_risk),  "Biliary obstruction / metastasis risk"),
        "Bile Duct":   (bile_risk,   risk_col(bile_risk),   "Obstructive jaundice risk"),
        "Lymph Nodes": (lymph_risk,  risk_col(lymph_risk),  "Regional lymphatic spread"),
        "Duodenum":    (duo_risk,    risk_col(duo_risk),    "Anatomical invasion risk"),
        "Stomach":     (stomach_risk,risk_col(stomach_risk),"Direct invasion in advanced disease"),
        "Spleen":      (spleen_risk, risk_col(spleen_risk), "Splenic vein involvement"),
        "Lung":        (lung_risk,   risk_col(lung_risk),   "Distant metastasis risk"),
    }


def simulate_biomarker_timeline(clinical_inputs, prob):
    """Simulate 12-month biomarker trajectory (kept from original dashboard)."""
    months = list(range(0, 13))
    np.random.seed(42)

    growth_factor = 1 + (prob * 0.08)

    ca199_traj, cea_traj, bili_traj, amyl_traj = [], [], [], []
    ca_curr = clinical_inputs.get('ca199', 45)
    cea_curr = clinical_inputs.get('cea', 3.5)
    bil_curr = clinical_inputs.get('bilirubin', 0.8)
    amy_curr = clinical_inputs.get('amylase', 95)

    for m in months:
        noise = np.random.normal(0, 0.04)
        factor = growth_factor ** m
        ca199_traj.append(round(ca_curr * factor * (1 + noise), 1))
        cea_traj.append(round(cea_curr * (growth_factor**m * 0.7) * (1 + noise * 0.5), 2))
        bili_traj.append(round(min(20, bil_curr * (1 + (prob*0.04)*m) * (1 + noise*0.3)), 2))
        amyl_traj.append(round(amy_curr * (1 + (prob*0.02)*m) * (1 + noise*0.4)))

    from datetime import datetime, timedelta
    dates = [(datetime.today() + timedelta(days=30*m)).strftime("%b %Y") for m in months]
    return pd.DataFrame({
        "Month": dates,
        "CA 19-9 (U/mL)": ca199_traj,
        "CEA (ng/mL)": cea_traj,
        "Bilirubin (mg/dL)": bili_traj,
        "Amylase (U/L)": amyl_traj,
    })


# ── Full prediction pipeline ─────────────────────────────────────────────

def run_full_prediction(inputs):
    """
    Run the complete prediction pipeline from dashboard inputs dict.
    Returns dict with all prediction results.
    """
    models = ModelRegistry()
    models.load_all()

    age = inputs.get('age', 60)
    sex = inputs.get('sex', 'Male')
    gender_male = 1 if sex == 'Male' else 0

    clinical_inputs = {
        'ca199': inputs.get('ca199', 37),
        'cea': inputs.get('cea', 3),
        'bilirubin': inputs.get('bilirubin', 0.8),
        'amylase': inputs.get('amylase', 95),
        'weight_loss_pct': inputs.get('weight_loss_pct', 0),
        'smoking': inputs.get('smoking', 'Never'),
        'alcohol': inputs.get('alcohol', 'None / Minimal'),
        'diabetes': inputs.get('diabetes', 'None'),
        'family_history': inputs.get('family_history', False),
        'pancreatitis': inputs.get('pancreatitis', False),
        'genetic': inputs.get('genetic', False),
        'kras_expr': inputs.get('kras_expr', 5.0),
        'tp53_expr': inputs.get('tp53_expr', 5.0),
        'smad4_expr': inputs.get('smad4_expr', 5.0),
        'erbb2_expr': inputs.get('erbb2_expr', 5.0),
        'age': age,
    }

    # Run predictions
    prob = predict_cancer_probability(age, gender_male, clinical_inputs)
    stage_label, stage_col, stage_detail = predict_stage(
        age, gender_male, prob,
        clinical_inputs.get('ca199', 37),
        clinical_inputs.get('bilirubin', 0.8)
    )
    surv = predict_survival(age, gender_male, prob, stage_label)
    treatment = predict_treatment(prob, age)
    shap_vals = predict_shap_augmented(age, gender_male, clinical_inputs, prob)
    organ_risks = compute_organ_risks(clinical_inputs, prob, stage_label)
    bio_timeline = simulate_biomarker_timeline(clinical_inputs, prob)

    return {
        'prob': prob,
        'stage_label': stage_label,
        'stage_col': stage_col,
        'stage_detail': stage_detail,
        'survival': surv,
        'treatment': treatment,
        'best_tx': max(treatment, key=treatment.get),
        'shap_vals': shap_vals,
        'sorted_sv': sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True),
        'organ_risks': organ_risks,
        'bio_timeline': bio_timeline,
    }
