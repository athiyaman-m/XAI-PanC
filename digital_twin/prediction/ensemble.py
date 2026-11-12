import numpy as np
import pandas as pd
from typing import Optional

from .detection_model import DetectionPredictor
from .stage_model import StagePredictor
from .survival_model import SurvivalPredictor
from ..explainability.shap_engine import ShapEngine


class PredictionEnsemble:
    def __init__(self):
        self.detector = DetectionPredictor()
        self.stager = StagePredictor()
        self.survival = SurvivalPredictor()
        self.shap_engine = ShapEngine()

    def predict_treatment(self, prob: float, age: int) -> dict:
        folfi = 0.5 + (0.15 if age < 65 else 0) + (0.08 if prob < 0.6 else 0)
        gem = 0.45 + max(0, (1 - prob) * 0.2)
        return {
            "FOLFIRINOX":                   float(np.clip(folfi, 0.1, 0.92)),
            "Gemcitabine + nab-paclitaxel": float(np.clip(gem, 0.1, 0.88)),
            "FOLFOX":                       float(np.clip(folfi * 0.85, 0.1, 0.80)),
            "Best Supportive Care":         float(np.clip(0.30 + prob * 0.15, 0.1, 0.55)),
        }

    def compute_organ_risks(self, clinical_inputs: dict, prob: float,
                            stage_label: str) -> dict:
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

    def simulate_biomarker_timeline(self, clinical_inputs: dict,
                                    prob: float) -> pd.DataFrame:
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
            cea_traj.append(round(cea_curr * (growth_factor ** m * 0.7) * (1 + noise * 0.5), 2))
            bili_traj.append(round(min(20, bil_curr * (1 + (prob * 0.04) * m) * (1 + noise * 0.3)), 2))
            amyl_traj.append(round(amy_curr * (1 + (prob * 0.02) * m) * (1 + noise * 0.4)))
        from datetime import datetime, timedelta
        dates = [(datetime.today() + timedelta(days=30 * m)).strftime("%b %Y") for m in months]
        return pd.DataFrame({
            "Month": dates, "CA 19-9 (U/mL)": ca199_traj,
            "CEA (ng/mL)": cea_traj, "Bilirubin (mg/dL)": bili_traj,
            "Amylase (U/L)": amyl_traj,
        })

    def run_full_prediction(self, inputs: dict) -> dict:
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

        prob = self.detector.predict_with_biomarkers(age, gender_male, clinical_inputs)
        stage_label, stage_col, stage_detail = self.stager.predict(
            age, gender_male, prob,
            clinical_inputs.get('ca199', 37),
            clinical_inputs.get('bilirubin', 0.8),
        )
        surv = self.survival.predict(age, gender_male, prob, stage_label)
        treatment = self.predict_treatment(prob, age)
        shap_vals = self.shap_engine.predict_shap_augmented(age, gender_male, clinical_inputs, prob)
        organ_risks = self.compute_organ_risks(clinical_inputs, prob, stage_label)
        bio_timeline = self.simulate_biomarker_timeline(clinical_inputs, prob)

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
