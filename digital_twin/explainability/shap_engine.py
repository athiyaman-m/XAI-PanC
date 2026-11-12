import numpy as np
import xgboost as xgb
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from ..config.config import MODELS_DIR, BASE_DIR


class ShapEngine:
    def __init__(self):
        self._model = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        import sys
        sys.path.insert(0, BASE_DIR)
        from backend.prediction_engine import ModelRegistry
        self._registry = ModelRegistry()
        self._registry.load_all()
        self._model = self._registry.clinical_detection
        self._loaded = True

    def predict_shap(self, age: int, gender_male: int) -> dict:
        self.load()
        booster = self._model.get_booster()
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

    def predict_shap_augmented(self, age: int, gender_male: int,
                                clinical_inputs: dict, prob: float) -> dict:
        base_shap = self.predict_shap(age, gender_male)
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
            0.10 if "New-onset" in dm else 0.03 if "Long" in dm else 0.0, 4)
        smoke = clinical_inputs.get('smoking', 'Never')
        heuristic["Smoking History"] = round(
            0.08 if smoke == "Current" else 0.04 if smoke == "Former" else 0.0, 4)
        heuristic["Family History"] = round(0.07 if clinical_inputs.get('family_history') else 0.0, 4)
        heuristic["Pancreatitis History"] = round(0.06 if clinical_inputs.get('pancreatitis') else 0.0, 4)
        return {**heuristic, **base_shap}
