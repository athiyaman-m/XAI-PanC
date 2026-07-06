"""
digital_twin — Next-generation Clinical Digital Twin platform.

Modules
-------
ai_twin_engine      : DigitalTwinEngine orchestrator
organ_models        : Organ / Lesion data objects
anatomy_loader      : 3D mesh loading (STL/OBJ/GLTF/GLB + procedural fallback)
organ_renderer      : Multi-layer Plotly 3D rendering (7 visualisation modes)
organ_animation     : Spread paths, pulsing glow, particle effects
lesion_mapper       : Localised tumour hotspot rendering
disease_progression : Timeline simulation (0-36 months)
biomarker_mapper    : Biomarker-to-organ linkage
clinical_overlay    : Hover tooltips + patient side panel
"""

from .ai_twin_engine import DigitalTwinEngine
from .organ_models import Organ, Lesion
from .organ_renderer import RENDER_MODES
from .clinical_overlay import organ_hover_html, patient_panel_html

__all__ = [
    "DigitalTwinEngine",
    "Organ",
    "Lesion",
    "RENDER_MODES",
    "organ_hover_html",
    "patient_panel_html",
]
