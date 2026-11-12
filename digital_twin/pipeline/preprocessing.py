import os
import pickle
import numpy as np
from typing import Optional
from ..config.config import MODELS_DIR


class PreprocessingPipeline:
    def __init__(self):
        self.scaler = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        path = os.path.join(MODELS_DIR, "scaler.pkl")
        if os.path.isfile(path):
            with open(path, 'rb') as f:
                self.scaler = pickle.load(f)
        self._loaded = True

    def scale(self, features: np.ndarray) -> np.ndarray:
        self.load()
        if self.scaler is not None:
            return self.scaler.transform(features)
        return features

    def normalize_biomarkers(self, value: float, biomarker: str) -> float:
        thresholds = {
            "ca199": 37, "cea": 5, "bilirubin": 1.2,
            "amylase": 110, "weight_loss_pct": 5,
        }
        th = thresholds.get(biomarker, 0)
        if th == 0 or value <= th:
            return 0.0
        return min(1.0, (value - th) / (th * 3))
