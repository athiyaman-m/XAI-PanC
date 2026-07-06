"""
ai_twin_engine.py — DigitalTwinEngine: the central orchestrator.

Receives patient data + AI predictions and produces a dynamic
anatomical state with health scores, risks, lesions, colours,
and progression simulation.

All rendering is delegated to organ_renderer.py.
"""

import numpy as np
import plotly.graph_objects as go
from copy import deepcopy
from typing import Optional

from .organ_models import Organ, Lesion
from .anatomy_loader import load_organ_mesh
from .biomarker_mapper import get_organ_biomarkers, compute_biomarker_impact
from .disease_progression import simulate_progression, interpolate_state
from .organ_renderer import RENDER_MODES, _organ_to_mesh3d, _torso_outline

import warnings
warnings.filterwarnings("ignore")


# ── Default organ definitions ─────────────────────────────────

DEFAULT_ORGANS = [
    "Pancreas", "Liver", "Gallbladder", "Bile Duct",
    "Duodenum", "Stomach", "Spleen", "Lungs",
    "Lymph Nodes", "Heart", "Kidneys",
]

# Anatomical centres for spread / pulse effects
# Coordinate system: x=left-right (-=left), y=front-back (+=front), z=up-down (+=up)
ORGAN_CENTRES = {
    "Pancreas":     (0.0, -0.10, 0.05),
    "Liver":        (0.55, 0.18, 0.55),
    "Gallbladder":  (0.78, 0.32, 0.28),
    "Bile Duct":    (0.50, 0.24, 0.12),
    "Duodenum":     (0.38, 0.04, -0.04),
    "Stomach":      (-0.26, 0.26, 0.12),
    "Spleen":       (-1.02, -0.08, 0.35),
    "Lungs":        (0.0, 0.04, 1.15),
    "Lymph Nodes":  (0.0, -0.08, 0.08),
    "Heart":        (-0.10, 0.22, 1.00),
    "Kidneys":      (0.0, -0.18, 0.0),
}

# Spread pathways (for disease progression)
SPREAD_PATHS = {
    "Pancreas":  ["Lymph Nodes", "Liver", "Duodenum", "Stomach"],
    "Liver":     ["Lungs", "Bile Duct"],
    "Duodenum":  ["Stomach", "Lymph Nodes"],
    "Lymph Nodes": ["Lungs", "Liver", "Pancreas"],
}


class DigitalTwinEngine:
    """
    Central orchestrator for the Clinical Digital Twin.

    Usage:
        engine = DigitalTwinEngine()
        engine.load_patient(clinical_inputs, predictions)
        engine.update_all()
        fig = engine.display(mode='Risk Heatmap', progression_months=0)
    """

    def __init__(self):
        self.organs: dict[str, Organ] = {}
        self.patient_data: dict = {}
        self.predictions: dict = {}
        self.overall_health: float = 100.0
        self.overall_risk: float = 0.0
        self.progression_base: dict[str, Organ] = {}  # snapshot at t=0
        self.current_mode: str = "Risk Heatmap"
        self.progression_months: int = 0
        self.treatment_name: str = ""
        self.treatment_response: float = 0.0
        self._baseline_state: dict = {}

    # ── Patient loading ────────────────────────────────────────

    def load_patient(self, clinical_inputs: dict, predictions: dict):
        """
        Initialise or update the digital twin from clinical data
        and AI prediction outputs.

        Parameters
        ----------
        clinical_inputs : dict — raw sidebar/CSV inputs
        predictions : dict — output from prediction_engine.run_full_prediction()
        """
        self.patient_data = dict(clinical_inputs)
        self.predictions = dict(predictions)

        prob = predictions.get('prob', 0.0)
        stage_label = predictions.get('stage_label', 'Normal')
        organ_risks = predictions.get('organ_risks', {})
        surv = predictions.get('survival', {})
        shap_vals = predictions.get('shap_vals', {})
        treatment = predictions.get('treatment', {})

        # Create / update organs
        for name in DEFAULT_ORGANS:
            risk_info = organ_risks.get(name, (0, "#2ecc71", ""))
            risk_pct = risk_info[0] if isinstance(risk_info, (list, tuple)) and len(risk_info) >= 1 else 0
            risk_col = risk_info[1] if isinstance(risk_info, (list, tuple)) and len(risk_info) >= 2 else "#2ecc71"
            expl = risk_info[2] if isinstance(risk_info, (list, tuple)) and len(risk_info) >= 3 else ""

            organ = self.organs.get(name)
            if organ is None:
                organ = Organ(name=name)
                # Pre-load mesh once
                organ.plotly_mesh = load_organ_mesh(name)
                self.organs[name] = organ

            # Risk / Health
            organ.risk_score = float(risk_pct)
            organ.health_score = max(0.0, 100.0 - risk_pct * 1.15)
            organ.disease_stage = self._stage_from_risk(risk_pct)
            organ.colour = organ.colour_for_risk()
            organ.opacity = 0.75 + (risk_pct / 100.0) * 0.20
            organ.explanation = expl

            # Biomarkers
            for bname in get_organ_biomarkers(name):
                raw_val = clinical_inputs.get(bname.split(" (")[0].lower().replace(" ", "_"), 0)
                organ.biomarkers[bname] = float(raw_val) if raw_val else 0.0

            # SHAP
            for shap_key, shap_val in shap_vals.items():
                if name.lower() in shap_key.lower():
                    organ.shap_values[shap_key] = float(shap_val)

            # Tumour
            if name == "Pancreas" and prob > 0.25:
                organ.tumour_size = prob * 3.5  # ~cm
                lesion_pos = ORGAN_CENTRES.get(name, (0, 0, 0))
                organ.lesions = [
                    Lesion(
                        position=lesion_pos,
                        size=0.05 + prob * 0.12,
                        opacity=0.5 + prob * 0.4,
                        glow_intensity=prob,
                        confidence=prob,
                        is_metastasis=False,
                    )
                ]

            # Spread
            if name in SPREAD_PATHS and risk_pct > 35:
                for target in SPREAD_PATHS[name]:
                    target_organ = self.organs.get(target)
                    target_risk = target_organ.risk_score if target_organ else 0
                    if target_risk > 20 and target not in organ.spread_regions:
                        organ.spread_regions.append(target)

            # Stage-appropriate recommendations
            organ.recommendation = self._recommendation(name, risk_pct, stage_label)

        # Overall metrics
        self._compute_overall_metrics()
        self._baseline_state = {n: deepcopy(o) for n, o in self.organs.items()}
        self.progression_base = deepcopy(self._baseline_state)

    # ── Updates ────────────────────────────────────────────────

    def update_all(self):
        """Recompute all organ states from current patient + predictions."""
        if not self.predictions:
            return
        self.load_patient(self.patient_data, self.predictions)

    def update_progression(self, months: int):
        """
        Advance the digital twin by `months` into the future.

        Uses the baseline (t=0) state and simulates progression.
        """
        self.progression_months = months
        if months <= 0:
            self.organs = deepcopy(self._baseline_state)
            self._compute_overall_metrics()
            return

        for name in DEFAULT_ORGANS:
            if name not in self.progression_base:
                continue
            base = self.progression_base[name]
            prog = simulate_progression(
                base, months,
                base_prob=self.predictions.get('prob', 0.5),
            )
            self.organs[name] = prog

        self._compute_overall_metrics()

    def apply_treatment(self, treatment_name: str, response_prob: float):
        """
        Apply a treatment simulation.

        Improves health and reduces risk proportionally to response_prob.
        """
        self.treatment_name = treatment_name
        self.treatment_response = response_prob

        for name, organ in self.organs.items():
            health_boost = response_prob * 30
            risk_reduction = response_prob * 25
            organ.health_score = min(100, organ.health_score + health_boost)
            organ.risk_score = max(0, organ.risk_score - risk_reduction)
            organ.colour = organ.colour_for_risk()
            if response_prob > 0.5:
                organ.lesions = []
                organ.tumour_size *= max(0, 1 - response_prob * 0.8)

        self._compute_overall_metrics()

    def highlight_shap_feature(self, feature_name: str):
        """
        Highlight organs affected by a given SHAP feature.

        Affected organs get a glow overlay.
        """
        for name, organ in self.organs.items():
            if feature_name.lower() in name.lower():
                organ.opacity = 0.95
            else:
                has_match = any(
                    feature_name.lower() in k.lower()
                    for k in organ.shap_values
                )
                organ.opacity = 0.95 if has_match else 0.30

    def reset_highlight(self):
        """Reset all highlights to default opacity."""
        for name, organ in self.organs.items():
            organ.opacity = 0.75 + (organ.risk_score / 100.0) * 0.20

    # ── Rendering ──────────────────────────────────────────────

    def display(self, mode: str = "Risk Heatmap") -> go.Figure:
        """
        Render the current digital twin state using the chosen mode.

        Parameters
        ----------
        mode : str — one of RENDER_MODES keys

        Returns
        -------
        go.Figure
        """
        self.current_mode = mode
        renderer = RENDER_MODES.get(mode, RENDER_MODES["Risk Heatmap"])

        if mode == "Treatment Response":
            traces = renderer(self.organs, self.treatment_name, self.treatment_response)
        else:
            traces = renderer(self.organs)

        # Legend annotations
        traces.extend(self._legend_traces())

        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False, showgrid=False, range=[-2.2, 2.2]),
                yaxis=dict(visible=False, showgrid=False, range=[-1.0, 1.0]),
                zaxis=dict(visible=False, showgrid=False, range=[-2.4, 2.4]),
                bgcolor='rgba(0,0,0,0)',
                camera=dict(
                    eye=dict(x=2.8, y=-2.2, z=1.5),
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0.3),
                ),
                aspectmode='manual',
                aspectratio=dict(x=1.3, y=0.65, z=1.2),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0),
            height=600,
            showlegend=False,
        )
        return fig

    # ── Internal ───────────────────────────────────────────────

    def _compute_overall_metrics(self):
        """Compute overall health and risk from all organs."""
        if not self.organs:
            self.overall_health = 100.0
            self.overall_risk = 0.0
            return
        scores = [(o.health_score, o.risk_score) for o in self.organs.values()]
        self.overall_health = np.mean([s[0] for s in scores])
        self.overall_risk = np.mean([s[1] for s in scores])

    def _stage_from_risk(self, risk: float) -> str:
        if risk >= 70:
            return "Stage IV"
        if risk >= 50:
            return "Stage III"
        if risk >= 35:
            return "Stage II"
        if risk >= 20:
            return "Stage I"
        return "Normal"

    def _recommendation(self, organ_name: str, risk: float,
                        ai_stage: str) -> str:
        if risk >= 70:
            return f"Urgent: {organ_name} requires immediate intervention"
        if risk >= 45:
            return f"Monitor {organ_name} closely — consider further imaging"
        if risk >= 20:
            return f"Routine surveillance for {organ_name}"
        return f"{organ_name} within normal limits"

    def _legend_traces(self) -> list:
        """Risk legend as 3D text annotations."""
        items = [
            (-1.8, -0.7, -2.15, '● Low <20%',   '#2ecc71'),
            (-1.8, -0.7, -2.25, '● Mild 20-44%', '#f1c40f'),
            (-1.8, -0.7, -2.35, '● Moderate 45-69%', '#f39c12'),
            (-1.8, -0.7, -2.45, '● High ≥70%',  '#dc3232'),
        ]
        traces = []
        for lx, ly, lz, lt, lc in items:
            traces.append(go.Scatter3d(
                x=[lx], y=[ly], z=[lz],
                mode='text', text=[lt],
                textfont=dict(size=9, color=lc, family='Inter, sans-serif'),
                hoverinfo='skip', showlegend=False,
            ))
        return traces
