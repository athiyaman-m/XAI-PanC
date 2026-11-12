import numpy as np


class SurvivalPredictor:
    def __init__(self):
        self._model = None

    def load(self):
        if self._model is not None:
            return
        import sys, os
        from ..config.config import BASE_DIR
        sys.path.insert(0, BASE_DIR)
        from backend.prediction_engine import ModelRegistry
        self._registry = ModelRegistry()
        self._registry.load_all()
        self._model = self._registry.clinical_survival

    def predict(self, age: int, gender_male: int, prob: float,
                stage_label: str) -> dict:
        self.load()
        feat = np.array([[age, gender_male]], dtype=float)
        risk_score = float(self._model.predict(feat)[0])
        base_hazard = risk_score / 10.0
        if 'III' in stage_label or 'IV' in stage_label:
            base_hazard += 1.5
        elif 'II' in stage_label:
            base_hazard += 0.5
        base_hazard += prob * 2.0
        base_hazard += max(0, (age - 60) / 40)
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
