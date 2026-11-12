import numpy as np
from typing import Optional


class DetectionPredictor:
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
        self._model = self._registry.clinical_detection

    def predict(self, age: int, gender_male: int) -> float:
        self.load()
        age_score = max(0.0, (age - 40) / 80.0) * 0.20
        if age >= 70:
            age_score += 0.05
        elif age >= 60:
            age_score += 0.02
        return age_score

    def predict_proba(self, age: int, gender_male: int) -> float:
        return self.predict(age, gender_male)

    def predict_with_biomarkers(self, age: int, gender_male: int,
                                 clinical_inputs: dict) -> float:
        age_score = self.predict(age, gender_male)
        bio_score = 0.0
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
