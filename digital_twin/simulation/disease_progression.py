import numpy as np
from copy import deepcopy
from ..anatomy.organ_models import Organ, Lesion


def simulate_progression(organ: Organ, months: int,
                         base_prob: float = 0.5,
                         growth_rate: float = 0.08) -> Organ:
    prog = deepcopy(organ)
    if months <= 0:
        return prog
    growth = np.exp(growth_rate * months) - 1.0
    prog.tumour_size = organ.tumour_size + growth * 0.15
    risk_delta = (1.0 - organ.risk_score / 100.0) * 0.3 * (1 - np.exp(-months * 0.12))
    prog.risk_score = min(98.0, organ.risk_score + risk_delta * 50)
    health_loss = risk_delta * 35
    prog.health_score = max(2.0, organ.health_score - health_loss)
    if prog.risk_score < 20:
        prog.disease_stage = "Normal"
    elif prog.risk_score < 35:
        prog.disease_stage = "Stage I"
    elif prog.risk_score < 50:
        prog.disease_stage = "Stage II"
    elif prog.risk_score < 70:
        prog.disease_stage = "Stage III"
    else:
        prog.disease_stage = "Stage IV"
    new_lesions = []
    for les in organ.lesions:
        grown = Lesion(
            position=les.position,
            size=les.size * (1 + 0.05 * months),
            opacity=min(1.0, les.opacity + 0.01 * months),
            glow_intensity=min(1.0, les.glow_intensity + 0.02 * months),
            confidence=min(1.0, les.confidence + 0.015 * months),
            is_metastasis=les.is_metastasis or months > 6,
        )
        new_lesions.append(grown)
    prog.lesions = new_lesions
    spread_map = {
        "Pancreas": ["Lymph Nodes", "Liver", "Duodenum"],
        "Liver": ["Lungs", "Bile Duct"],
        "Lymph Nodes": ["Lungs", "Liver"],
    }
    existing = set(prog.spread_regions)
    for region in spread_map.get(prog.name, []):
        if region not in existing and prog.risk_score > 40 + months * 1.5:
            prog.spread_regions.append(region)
    prog.colour = prog.colour_for_risk()
    return prog


def interpolate_state(organ_a: Organ, organ_b: Organ, t: float) -> Organ:
    interp = deepcopy(organ_a)
    interp.health_score = organ_a.health_score + (organ_b.health_score - organ_a.health_score) * t
    interp.risk_score = organ_a.risk_score + (organ_b.risk_score - organ_a.risk_score) * t
    interp.tumour_size = organ_a.tumour_size + (organ_b.tumour_size - organ_a.tumour_size) * t
    interp.opacity = organ_a.opacity + (organ_b.opacity - organ_a.opacity) * t
    interp.colour = _lerp_colour(organ_a.colour, organ_b.colour, t)
    return interp


def _lerp_colour(c1: str, c2: str, t: float) -> str:
    def hex_to_rgb(h):
        h = h.lstrip('#')
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))
    def rgb_to_hex(r, g, b):
        return f'#{int(r):02x}{int(g):02x}{int(b):02x}'
    r1, g1, b1 = hex_to_rgb(c1)
    r2, g2, b2 = hex_to_rgb(c2)
    r = r1 + (r2 - r1) * t
    g = g1 + (g2 - g1) * t
    b = b1 + (b2 - b1) * t
    return rgb_to_hex(r, g, b)
