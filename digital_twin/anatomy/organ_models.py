from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Lesion:
    position: tuple[float, float, float]
    size: float
    opacity: float = 0.75
    glow_intensity: float = 0.5
    bounding_region: tuple = ()
    confidence: float = 0.0
    is_metastasis: bool = False
    layers: int = 3
    seed: int = 7


@dataclass
class Organ:
    name: str
    plotly_mesh: Optional[dict] = None
    mesh_path: Optional[str] = None
    health_score: float = 100.0
    risk_score: float = 0.0
    disease_stage: str = "Normal"
    affected_regions: list[str] = field(default_factory=list)
    lesions: list[Lesion] = field(default_factory=list)
    colour: str = "#2ecc71"
    opacity: float = 0.85
    clinical_params: dict[str, float] = field(default_factory=dict)
    biomarkers: dict[str, float] = field(default_factory=dict)
    shap_values: dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    recommendation: str = ""
    tumour_size: float = 0.0
    spread_regions: list[str] = field(default_factory=list)

    def colour_for_risk(self) -> str:
        r = self.risk_score
        if r >= 70:
            return "#dc3232"
        if r >= 45:
            return "#f39c12"
        if r >= 20:
            return "#f1c40f"
        return "#2ecc71"

    def status_text(self) -> str:
        if self.risk_score >= 70:
            return "Critical"
        if self.risk_score >= 45:
            return "Moderate"
        if self.risk_score >= 20:
            return "Mild"
        return "Healthy"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "health": round(self.health_score, 1),
            "risk": round(self.risk_score, 1),
            "stage": self.disease_stage,
            "status": self.status_text(),
            "lesions": len(self.lesions),
            "biomarkers": dict(self.biomarkers),
        }

    def copy_state(self) -> "Organ":
        import copy
        return copy.copy(self)
