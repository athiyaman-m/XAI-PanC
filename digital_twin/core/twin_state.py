from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime
import numpy as np

from ..anatomy.organ_models import Organ
from ..pipeline.patient_profile import PatientProfile


@dataclass
class TwinState:
    patient: PatientProfile
    prob: float = 0.0
    stage_label: str = "Non-malignant"
    stage_colour: str = "#3498db"
    stage_detail: str = ""
    risk_group: str = "Low Risk"
    risk_score: float = 0.0
    overall_health: float = 100.0
    overall_risk: float = 0.0
    median_survival_months: float = 36.0
    confidence: str = "Moderate"
    organs: dict[str, Organ] = field(default_factory=dict)
    shap_values: dict = field(default_factory=dict)
    top_features: list = field(default_factory=list)
    important_biomarkers: list = field(default_factory=list)
    affected_organs: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    history: list = field(default_factory=list)

    def update_from_prediction(self, result: dict):
        self.prob = result.get('prob', self.prob)
        self.stage_label = result.get('stage_label', self.stage_label)
        self.stage_colour = result.get('stage_col', self.stage_colour)
        self.stage_detail = result.get('stage_detail', self.stage_detail)
        surv = result.get('survival', {})
        self.risk_group = surv.get('risk_group', self.risk_group)
        self.risk_score = surv.get('risk_score', self.risk_score)
        self.median_survival_months = surv.get('median_months', self.median_survival_months)
        shap = result.get('shap_vals', {})
        self.shap_values = shap
        self.top_features = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        self.affected_organs = list(result.get('organ_risks', {}).keys())
        conf = abs(self.prob - 0.5)
        self.confidence = "Very High" if conf > 0.3 else "Moderate" if conf > 0.15 else "Low"
        self.last_updated = datetime.now().isoformat()
        self.history.append({
            "timestamp": self.last_updated,
            "prob": self.prob,
            "stage": self.stage_label,
            "risk": self.risk_score,
        })

    def update_organ_scores(self, organs: dict[str, Organ]):
        self.organs = organs
        scores = [(o.health_score, o.risk_score) for o in organs.values()]
        if scores:
            self.overall_health = float(np.mean([s[0] for s in scores]))
            self.overall_risk = float(np.mean([s[1] for s in scores]))
