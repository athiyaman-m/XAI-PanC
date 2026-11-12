class StagePredictor:
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
        self._model = self._registry.clinical_stage

    def predict(self, age: int, gender_male: int, prob: float = 0.0,
                ca199: float = 37, bilirubin: float = 0.8) -> tuple:
        if prob > 0.80 and ca199 > 500:
            return "Stage IV", "#e74c3c", "Distant metastasis likely \u2014 systemic chemotherapy indicated"
        elif prob > 0.65 and ca199 > 200:
            return "Stage III", "#e67e22", "Locally advanced, unresectable \u2014 chemoradiotherapy"
        elif prob > 0.50 and ca199 > 100:
            return "Stage II", "#f39c12", "Regional spread \u2014 consider neoadjuvant therapy"
        elif prob > 0.35:
            return "Stage I", "#2ecc71", "Localised \u2014 surgical resection may be feasible"
        else:
            return "Non-malignant", "#3498db", "No strong malignancy evidence \u2014 continue monitoring"

    def predict_proba(self, age: int, gender_male: int, prob: float = 0.0,
                      ca199: float = 37, bilirubin: float = 0.8) -> tuple:
        return self.predict(age, gender_male, prob, ca199, bilirubin)
