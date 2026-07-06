"""
organ_models.py — Organ data objects with health/risk/stage/mesh state.

Every organ is an independent object carrying its own:
  - 3D mesh data (procedural fallback or loaded from file)
  - Health score, risk score, disease stage
  - Lesions, affected regions, AI explanations
  - Biomarker and SHAP linkages

This decouples organ state from rendering logic.
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Lesion:
    """Localized tumour hotspot — never colour an entire organ red."""
    position: tuple[float, float, float]  # (x, y, z) relative to organ centre
    size: float                           # radius in world units
    opacity: float = 0.75
    glow_intensity: float = 0.5
    bounding_region: tuple = ()           # (dx, dy, dz) for future segmentation mask
    confidence: float = 0.0
    is_metastasis: bool = False


@dataclass
class Organ:
    """Single anatomical organ with full clinical state."""

    name: str

    # --- Mesh ---
    plotly_mesh: Optional[dict] = None        # {x, y, z} arrays for procedural mesh
    mesh_path: Optional[str] = None           # File path (STL/OBJ/GLTF/GLB) for future loader

    # --- Health scores (0-100) ---
    health_score: float = 100.0                # 100 = fully healthy
    risk_score: float = 0.0                    # 0 = no risk

    # --- Disease state ---
    disease_stage: str = "Normal"
    affected_regions: list[str] = field(default_factory=list)

    # --- Lesions ---
    lesions: list[Lesion] = field(default_factory=list)

    # --- Appearance ---
    colour: str = "#2ecc71"                    # dynamic, set by engine
    opacity: float = 0.85

    # --- Clinical data ---
    clinical_params: dict[str, float] = field(default_factory=dict)
    biomarkers: dict[str, float] = field(default_factory=dict)
    shap_values: dict[str, float] = field(default_factory=dict)
    explanation: str = ""
    recommendation: str = ""

    # --- Progression ---
    tumour_size: float = 0.0                   # cm, 0 = no tumour
    spread_regions: list[str] = field(default_factory=list)

    def colour_for_risk(self) -> str:
        """Map health score to colour (green → yellow → orange → red)."""
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
        """Return a shallow copy for interpolation/progression."""
        import copy
        return copy.copy(self)
