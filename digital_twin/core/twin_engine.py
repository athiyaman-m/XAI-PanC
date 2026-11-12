import numpy as np
import plotly.graph_objects as go
from copy import deepcopy
from typing import Optional
import hashlib
import json

from ..anatomy.organ_models import Organ, Lesion
from ..anatomy.anatomy_loader import load_organ_mesh
from ..explainability.biomarker_mapper import get_organ_biomarkers
from ..simulation.disease_progression import simulate_progression, interpolate_state
from ..anatomy.organ_renderer import RENDER_MODES, _organ_to_mesh3d, _torso_outline
from ..simulation.compartmental_twin import (
    DAYS_PER_MONTH, PancreaticTwinModel, PancreaticTwinState,
    PancreaticTwinParams, Intervention, fit_twin_parameters,
    compute_twin_metrics, forecast_compartment_trajectory,
)
from ..config.config import DEFAULT_ORGANS, ORGAN_CENTRES, SPREAD_PATHS
from .twin_state import TwinState

CT_TWIN_MODES = ("Real CT Twin",)


class DigitalTwinEngine:
    def __init__(self):
        self.organs: dict[str, Organ] = {}
        self.patient_data: dict = {}
        self.predictions: dict = {}
        self.overall_health: float = 100.0
        self.overall_risk: float = 0.0
        self.progression_base: dict[str, Organ] = {}
        self.current_mode: str = "Risk Heatmap"
        self.progression_months: int = 0
        self.treatment_name: str = ""
        self.treatment_response: float = 0.0
        self._baseline_state: dict = {}
        self._fig_cache: dict = {}
        self.compartmental_state: PancreaticTwinState | None = None
        self.compartmental_params: PancreaticTwinParams | None = None
        self.forecast_data: dict | None = None
        self.forecast_days: int = 360
        self.twin_state: TwinState | None = None
        self.ct_twin_meta: dict | None = None
        self.ct_twin_case: str | None = None

    def load_patient(self, clinical_inputs: dict, predictions: dict):
        self.patient_data = dict(clinical_inputs)
        self.predictions = dict(predictions)
        prob = predictions.get('prob', 0.0)
        stage_label = predictions.get('stage_label', 'Normal')
        organ_risks = predictions.get('organ_risks', {})
        surv = predictions.get('survival', {})
        shap_vals = predictions.get('shap_vals', {})
        treatment = predictions.get('treatment', {})

        for name in DEFAULT_ORGANS:
            risk_info = organ_risks.get(name, (0, "#2ecc71", ""))
            risk_pct = risk_info[0] if isinstance(risk_info, (list, tuple)) and len(risk_info) >= 1 else 0
            risk_col = risk_info[1] if isinstance(risk_info, (list, tuple)) and len(risk_info) >= 2 else "#2ecc71"
            expl = risk_info[2] if isinstance(risk_info, (list, tuple)) and len(risk_info) >= 3 else ""

            organ = self.organs.get(name)
            if organ is None:
                organ = Organ(name=name)
                organ.plotly_mesh = load_organ_mesh(name)
                self.organs[name] = organ

            organ.risk_score = float(risk_pct)
            organ.health_score = max(0.0, 100.0 - risk_pct * 1.15)
            organ.disease_stage = self._stage_from_risk(risk_pct)
            organ.colour = organ.colour_for_risk()
            organ.opacity = 0.75 + (risk_pct / 100.0) * 0.20
            organ.explanation = expl

            for bname in get_organ_biomarkers(name):
                raw_val = clinical_inputs.get(bname.split(" (")[0].lower().replace(" ", "_"), 0)
                organ.biomarkers[bname] = float(raw_val) if raw_val else 0.0

            for shap_key, shap_val in shap_vals.items():
                if name.lower() in shap_key.lower():
                    organ.shap_values[shap_key] = float(shap_val)

            if name == "Pancreas" and prob > 0.25:
                organ.tumour_size = prob * 3.5
                lesion_pos = ORGAN_CENTRES.get(name, (0, 0, 0))
                organ.lesions = [
                    Lesion(position=lesion_pos, size=0.05 + prob * 0.12,
                           opacity=0.5 + prob * 0.4, glow_intensity=prob,
                           confidence=prob, is_metastasis=False)
                ]

            if name in SPREAD_PATHS and risk_pct > 35:
                for target in SPREAD_PATHS[name]:
                    target_organ = self.organs.get(target)
                    target_risk = target_organ.risk_score if target_organ else 0
                    if target_risk > 20 and target not in organ.spread_regions:
                        organ.spread_regions.append(target)

            organ.recommendation = self._recommendation(name, risk_pct, stage_label)

        self._compute_overall_metrics()
        self._baseline_state = {n: deepcopy(o) for n, o in self.organs.items()}
        self.progression_base = deepcopy(self._baseline_state)
        self._init_compartmental_twin(clinical_inputs, predictions)

    def _init_compartmental_twin(self, clinical_inputs: dict, predictions: dict):
        prob = predictions.get('prob', 0.0)
        stage_label = predictions.get('stage_label', 'Normal')
        treatment = predictions.get('treatment', {})
        shap_vals = predictions.get('shap_vals', {})
        organ_risks = predictions.get('organ_risks', {})
        ca199 = float(clinical_inputs.get('ca199', 0)) or float(clinical_inputs.get('ca_19_9', 0))
        cea = float(clinical_inputs.get('cea', 0))
        bili = float(clinical_inputs.get('bilirubin', 0))
        kras = float(clinical_inputs.get('kras_expr', 0))
        tp53 = float(clinical_inputs.get('tp53_expr', 0))
        smad4 = float(clinical_inputs.get('smad4_expr', 0))
        age = int(clinical_inputs.get('age', 60))
        params = fit_twin_parameters(
            cancer_prob=prob, stage_label=stage_label,
            ca199_value=ca199, cea_value=cea, bilirubin_value=bili,
            treatment_responses=treatment, kras_expr=kras,
            tp53_expr=tp53, smad4_expr=smad4, age=age,
        )
        self.compartmental_params = params
        initial_tumor = prob * params.K * 0.6
        state = PancreaticTwinState(T=initial_tumor, CA=ca199, CE=cea, B=bili, D=0.0, I=0.1)
        self.compartmental_state = state
        self._update_forecast()

    def _update_forecast(self):
        if self.compartmental_state is None or self.compartmental_params is None:
            return
        interventions = []
        if self.treatment_name and self.treatment_response > 0:
            for d in range(1, self.forecast_days + 1, 7):
                interventions.append(Intervention(
                    day=d, compartment='D',
                    amount=self.treatment_response * 0.6,
                    label=f"{self.treatment_name} dose",
                ))
        self.forecast_data = forecast_compartment_trajectory(
            initial_state=self.compartmental_state,
            params=self.compartmental_params,
            days=self.forecast_days,
            interventions=interventions if interventions else None,
        )

    def run_compartmental_forecast(self, days: int = 360,
                                   what_if_params: dict | None = None) -> dict:
        if self.compartmental_params is None:
            return {}
        params = self.compartmental_params
        if what_if_params:
            p = PancreaticTwinParams()
            for key, val in what_if_params.items():
                if hasattr(p, key):
                    setattr(p, key, val)
            params = p
        return forecast_compartment_trajectory(
            initial_state=self.compartmental_state, params=params, days=days,
        )

    def compute_compartment_organ_risks(self) -> dict[str, float]:
        if self.forecast_data is None:
            return {}
        latest = len(self.forecast_data['risk']) - 1
        twin_risk = float(self.forecast_data['risk'][latest])
        tumor_vol = float(self.forecast_data['T'][latest])
        return {
            "Pancreas": min(100, twin_risk * 1.0 + 5),
            "Liver": min(100, twin_risk * 0.6 + tumor_vol * 0.5),
            "Bile Duct": min(100, twin_risk * 0.5 + tumor_vol * 0.8),
            "Duodenum": min(100, twin_risk * 0.4 + tumor_vol * 0.3),
            "Spleen": min(100, twin_risk * 0.2),
            "Stomach": min(100, twin_risk * 0.3),
            "Lymph Nodes": min(100, twin_risk * 0.7),
            "Lungs": min(100, twin_risk * 0.35),
            "Gallbladder": min(80, twin_risk * 0.3),
            "Heart": min(60, twin_risk * 0.15),
            "Kidneys": min(60, twin_risk * 0.15),
        }

    def update_all(self):
        if not self.predictions:
            return
        self.load_patient(self.patient_data, self.predictions)

    def update_progression(self, months: int):
        self.progression_months = months
        if months <= 0:
            self.organs = deepcopy(self._baseline_state)
            self._compute_overall_metrics()
            return
        for name in DEFAULT_ORGANS:
            if name not in self.progression_base:
                continue
            base = self.progression_base[name]
            prog = simulate_progression(base, months, base_prob=self.predictions.get('prob', 0.5))
            self.organs[name] = prog
        days = int(months * DAYS_PER_MONTH)
        if self.compartmental_state is not None and self.compartmental_params is not None:
            forecast = self.compartmental_state
            params = self.compartmental_params
            model = PancreaticTwinModel(forecast, params)
            model.update(days)
            twin_state = model.state
            twin_health, twin_risk, twin_stage, twin_tumour = compute_twin_metrics(twin_state, params)
            if "Pancreas" in self.organs:
                panc = self.organs["Pancreas"]
                blended_risk = 0.7 * twin_risk + 0.3 * panc.risk_score
                panc.risk_score = min(100, blended_risk)
                panc.health_score = max(0, 100 - panc.risk_score * 1.15)
                panc.tumour_size = twin_tumour
                panc.disease_stage = twin_stage
                panc.colour = panc.colour_for_risk()
            self.forecast_data = forecast_compartment_trajectory(
                initial_state=self.compartmental_state, params=self.compartmental_params, days=days,
            )
        self._compute_overall_metrics()

    def apply_treatment(self, treatment_name: str, response_prob: float):
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
        if self.compartmental_state is not None and self.compartmental_params is not None:
            self.compartmental_state.D = response_prob * 0.5
            self.compartmental_state.I = 0.1 + response_prob * 0.3
            self._update_forecast()

    def highlight_shap_feature(self, feature_name: str):
        for name, organ in self.organs.items():
            if feature_name.lower() in name.lower():
                organ.opacity = 0.95
            else:
                has_match = any(feature_name.lower() in k.lower() for k in organ.shap_values)
                organ.opacity = 0.95 if has_match else 0.30

    def reset_highlight(self):
        for name, organ in self.organs.items():
            organ.opacity = 0.75 + (organ.risk_score / 100.0) * 0.20

    def _cache_key(self, mode: str) -> str:
        state = {
            "mode": mode, "prog": self.progression_months,
            "tx": self.treatment_name, "resp": round(self.treatment_response, 3),
            "organs": {n: (o.risk_score, o.health_score, o.opacity, len(o.lesions))
                       for n, o in self.organs.items()},
        }
        raw = json.dumps(state, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def display(self, mode: str = "Risk Heatmap") -> go.Figure:
        self.current_mode = mode
        if mode in CT_TWIN_MODES:
            from ..anatomy.real_ct_twin import render_real_ct_twin
            fig, meta = render_real_ct_twin(case=self.ct_twin_case)
            self.ct_twin_meta = meta
            return fig
        ckey = self._cache_key(mode)
        if ckey in self._fig_cache:
            return self._fig_cache[ckey]
        renderer = RENDER_MODES.get(mode, RENDER_MODES["Risk Heatmap"])
        if mode == "Treatment Response":
            traces = renderer(self.organs, self.treatment_name, self.treatment_response)
        elif mode == "Compartmental Twin":
            traces = renderer(self.organs, forecast_data=self.forecast_data)
        else:
            traces = renderer(self.organs)
        traces.extend(self._legend_traces())
        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(
                xaxis=dict(visible=False, showgrid=False, range=[-2.4, 2.4]),
                yaxis=dict(visible=False, showgrid=False, range=[-1.2, 1.2]),
                zaxis=dict(visible=False, showgrid=False, range=[-2.6, 2.6]),
                bgcolor='rgba(0,0,0,0)',
                camera=dict(eye=dict(x=0.0, y=-2.8, z=0.6),
                            up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0.15)),
                aspectmode='manual', aspectratio=dict(x=1.0, y=0.5, z=1.5),
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, t=0, b=0), height=900, showlegend=False,
        )
        if len(self._fig_cache) > 10:
            oldest = next(iter(self._fig_cache))
            del self._fig_cache[oldest]
        self._fig_cache[ckey] = fig
        return fig

    def clear_cache(self):
        self._fig_cache.clear()

    def _compute_overall_metrics(self):
        if not self.organs:
            self.overall_health = 100.0
            self.overall_risk = 0.0
            return
        scores = [(o.health_score, o.risk_score) for o in self.organs.values()]
        self.overall_health = np.mean([s[0] for s in scores])
        self.overall_risk = np.mean([s[1] for s in scores])

    def _stage_from_risk(self, risk: float) -> str:
        if risk >= 70: return "Stage IV"
        if risk >= 50: return "Stage III"
        if risk >= 35: return "Stage II"
        if risk >= 20: return "Stage I"
        return "Normal"

    def _recommendation(self, organ_name: str, risk: float, ai_stage: str) -> str:
        if risk >= 70:
            return f"Urgent: {organ_name} requires immediate intervention"
        if risk >= 45:
            return f"Monitor {organ_name} closely \u2014 consider further imaging"
        if risk >= 20:
            return f"Routine surveillance for {organ_name}"
        return f"{organ_name} within normal limits"

    def _legend_traces(self) -> list:
        items = [
            (-1.8, -0.7, -2.15, '\u25cf Low <20%',   '#2ecc71'),
            (-1.8, -0.7, -2.25, '\u25cf Mild 20-44%', '#f1c40f'),
            (-1.8, -0.7, -2.35, '\u25cf Moderate 45-69%', '#f39c12'),
            (-1.8, -0.7, -2.45, '\u25cf High \u226570%',  '#dc3232'),
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
