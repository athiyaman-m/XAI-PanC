"""
digital_twin — AI-Powered Clinical Digital Twin Platform

Architecture:
    core/               Twin state, engine, simulation, recommendations
    pipeline/           Patient data, feature engineering, preprocessing
    prediction/         Detection, stage, survival, ensemble predictors
    anatomy/            Organ models, mesh generation, rendering, lesion mapping
    visualization/      Dashboard charts, organ animation
    explainability/     SHAP engine, biomarker mapping, clinical overlays
    simulation/         Disease progression, compartmental ODE, what-if analysis
    config/             Paths, constants, metadata

Usage:
    from digital_twin import DigitalTwinEngine, TwinState, PatientProfile
    from digital_twin import PredictionEnsemble, RENDER_MODES
"""

try:
    from .core.twin_engine import DigitalTwinEngine
    from .core.twin_state import TwinState
    from .core.simulation_engine import SimulationEngine
    from .core.recommendation_engine import RecommendationEngine
    from .pipeline.patient_profile import PatientProfile
    from .pipeline.data_pipeline import DataPipeline
    from .prediction.ensemble import PredictionEnsemble
except Exception:  # pragma: no cover - optional runtime dependencies
    DigitalTwinEngine = TwinState = SimulationEngine = RecommendationEngine = None
    PatientProfile = DataPipeline = PredictionEnsemble = None

try:
    from .anatomy.organ_models import Organ, Lesion
    from .anatomy.organ_renderer import RENDER_MODES, REALISTIC_ORGAN_COLORS, ORGAN_CENTRES
except Exception:  # pragma: no cover - optional runtime dependencies
    Organ = Lesion = None
    RENDER_MODES = REALISTIC_ORGAN_COLORS = ORGAN_CENTRES = None

try:
    from .explainability.clinical_overlay import organ_hover_html, patient_panel_html
    from .explainability.shap_engine import ShapEngine
except Exception:  # pragma: no cover - optional runtime dependencies
    organ_hover_html = patient_panel_html = None
    ShapEngine = None

try:
    from .simulation.compartmental_twin import (
        PancreaticTwinModel, PancreaticTwinState, PancreaticTwinParams,
        fit_twin_parameters, forecast_compartment_trajectory,
    )
    from .simulation.intervention_simulator import InterventionSimulator, WhatIfResult
except Exception:  # pragma: no cover - optional runtime dependencies
    PancreaticTwinModel = PancreaticTwinState = PancreaticTwinParams = None
    fit_twin_parameters = forecast_compartment_trajectory = None
    InterventionSimulator = WhatIfResult = None

try:
    from .visualization.charts import (
        gauge_chart, shap_chart, tx_chart, km_chart,
        plot_organ_radar, plot_biomarker_timeline,
    )
except Exception:  # pragma: no cover - optional runtime dependencies
    gauge_chart = shap_chart = tx_chart = km_chart = None
    plot_organ_radar = plot_biomarker_timeline = None

try:
    from .config.config import DEFAULT_ORGANS, ORGAN_CENTRES, SPREAD_PATHS, MODELS_DIR
except Exception:  # pragma: no cover - optional runtime dependencies
    DEFAULT_ORGANS = ORGAN_CENTRES = SPREAD_PATHS = MODELS_DIR = None

__all__ = [
    "DigitalTwinEngine", "TwinState", "SimulationEngine", "RecommendationEngine",
    "PatientProfile", "DataPipeline", "PredictionEnsemble",
    "Organ", "Lesion", "RENDER_MODES", "REALISTIC_ORGAN_COLORS",
    "organ_hover_html", "patient_panel_html", "ShapEngine",
    "PancreaticTwinModel", "PancreaticTwinState", "PancreaticTwinParams",
    "fit_twin_parameters", "forecast_compartment_trajectory",
    "InterventionSimulator", "WhatIfResult",
    "gauge_chart", "shap_chart", "tx_chart", "km_chart",
    "plot_organ_radar", "plot_biomarker_timeline",
    "DEFAULT_ORGANS", "ORGAN_CENTRES", "SPREAD_PATHS", "MODELS_DIR",
]
