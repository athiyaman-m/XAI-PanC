"""Deterministic biomarker-to-tumor geometry mapping.

Converts clinical biomarker values (CA 19-9, CEA, bilirubin, genetic markers,
etc.) into 3D tumor configuration parameters and per-organ disease burden
percentages.  The mapping is intentionally *not* an ML model — it is a
rule-based heuristic designed for < 1 ms latency so the browser receives an
instant response every time the user adjusts a slider.

The returned ``TumorConfig`` dict is structurally identical to the TypeScript
``TumorConfig`` type in ``syntheticModelEngine.ts``, ensuring seamless
frontend consumption.
"""
from __future__ import annotations

import math
from typing import Any


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * _clamp(t)


def map_biomarkers_to_tumor(bio: dict[str, Any]) -> dict[str, Any]:
    """Map biomarker values → tumor geometry config + organ burdens.

    Parameters
    ----------
    bio : dict
        Keys (all optional, sensible defaults applied):
        - ca199       : float  CA 19-9 in U/mL (normal ≤ 37)
        - cea         : float  CEA in ng/mL (normal ≤ 5)
        - bilirubin   : float  serum bilirubin in mg/dL (normal ≤ 1.2)
        - albumin     : float  serum albumin in g/dL (normal 3.5–5.5)
        - weight_loss : float  percentage body weight loss
        - kras        : bool   KRAS mutation detected
        - tp53        : bool   TP53 mutation detected
        - smad4       : bool   SMAD4 alteration detected
        - age         : int    patient age

    Returns
    -------
    dict with:
        tumor_config : dict   {sizeCm, position, invasionDepth, necrosisPct,
                               vascularInvasion, glowIntensity}
        organs       : dict   {OrganName: burden_pct}  (0–100)
        stage_hint   : str    approximate AJCC stage hint
        risk_score   : float  composite 0–100
    """
    ca199 = float(bio.get("ca199", 37))
    cea = float(bio.get("cea", 3))
    bilirubin = float(bio.get("bilirubin", 0.8))
    albumin = float(bio.get("albumin", 4.0))
    weight_loss = float(bio.get("weight_loss", 0))
    kras = bool(bio.get("kras", False))
    tp53 = bool(bio.get("tp53", False))
    smad4 = bool(bio.get("smad4", False))
    age = int(bio.get("age", 60))

    # ── Tumor size (cm) ──────────────────────────────────────────────────
    # Primary driver is CA 19-9.
    if ca199 <= 37:
        size_cm = _lerp(0.0, 0.8, ca199 / 37.0)
    elif ca199 <= 200:
        size_cm = _lerp(0.8, 2.0, (ca199 - 37) / 163.0)
    elif ca199 <= 1000:
        size_cm = _lerp(2.0, 4.0, (ca199 - 200) / 800.0)
    else:
        size_cm = _lerp(4.0, 7.0, _clamp((ca199 - 1000) / 4000.0))

    # CEA and weight loss secondarily enlarge the tumour.
    size_cm += _clamp(cea / 100.0) * 0.6
    size_cm += _clamp(weight_loss / 30.0) * 0.5

    # Mutations indicate more aggressive phenotype → slightly larger.
    mutation_count = sum([kras, tp53, smad4])
    size_cm += mutation_count * 0.15
    size_cm = _clamp(size_cm, 0.0, 8.0)

    # ── Invasion depth (0–1) ─────────────────────────────────────────────
    invasion = 0.0
    if cea > 5:
        invasion += _clamp((cea - 5) / 50.0) * 0.35
    if kras:
        invasion += 0.15
    if tp53:
        invasion += 0.15
    if smad4:
        invasion += 0.2
    invasion += _clamp(weight_loss / 30.0) * 0.15
    invasion = _clamp(invasion)

    # ── Necrosis percentage (0–1) ────────────────────────────────────────
    # Large, aggressive tumours develop central necrosis.
    necrosis = 0.0
    if size_cm > 3.0:
        necrosis = _clamp((size_cm - 3.0) / 5.0) * 0.6
    if tp53:
        necrosis += 0.1
    necrosis = _clamp(necrosis)

    # ── Vascular invasion (0–1) ──────────────────────────────────────────
    vasc = 0.0
    if ca199 > 200:
        vasc += _clamp((ca199 - 200) / 2000.0) * 0.4
    if smad4:
        vasc += 0.25
    if invasion > 0.4:
        vasc += 0.15
    vasc = _clamp(vasc)

    # ── Glow intensity (visual emphasis, 0–1) ────────────────────────────
    glow = _clamp(size_cm / 6.0) * 0.8 + invasion * 0.2
    glow = _clamp(glow, 0.05, 1.0)

    # ── Position (relative to pancreas center in the twin space) ─────────
    # High bilirubin → tumour near head (bile duct obstruction).
    # Otherwise default to body of pancreas.
    bili_t = _clamp(bilirubin / 8.0)
    pos_x = _lerp(0.0, 0.15, bili_t)     # shift toward head
    pos_y = _lerp(-0.08, -0.05, bili_t)
    pos_z = _lerp(0.05, 0.15, bili_t)

    # ── Per-organ disease burden (0–100) ─────────────────────────────────
    panc_burden = _clamp(size_cm / 6.0) * 70 + invasion * 20 + 10
    liver_burden = (
        _clamp(bilirubin / 10.0) * 40
        + _clamp(ca199 / 3000.0) * 30
        + (15 if size_cm > 3.0 and vasc > 0.3 else 0)
    )
    bile_burden = _clamp(bilirubin / 6.0) * 65 + (20 if bilirubin > 3 else 0)
    duo_burden = invasion * 50 + _clamp(size_cm / 5.0) * 25
    spleen_burden = vasc * 30 + _clamp(size_cm / 8.0) * 15
    lung_burden = (
        _clamp(ca199 / 3000.0) * 25
        + (15 if smad4 else 0)
        + (10 if size_cm > 4.0 else 0)
    )

    organs = {
        "Pancreas": round(min(panc_burden, 98), 1),
        "Liver": round(min(liver_burden, 95), 1),
        "Bile Duct": round(min(bile_burden, 92), 1),
        "Duodenum": round(min(duo_burden, 80), 1),
        "Spleen": round(min(spleen_burden, 70), 1),
        "Lungs": round(min(lung_burden, 60), 1),
    }

    # ── Composite risk score (0–100) ─────────────────────────────────────
    risk = (
        _clamp(size_cm / 6.0) * 35
        + invasion * 20
        + vasc * 15
        + _clamp(ca199 / 2000.0) * 15
        + mutation_count / 3.0 * 10
        + _clamp((age - 40) / 40.0) * 5
    )
    risk = round(min(risk * 100, 99), 1)

    # ── Stage hint ───────────────────────────────────────────────────────
    if size_cm < 0.5:
        stage = "Non-malignant"
    elif size_cm < 2.0 and invasion < 0.3:
        stage = "Stage I"
    elif size_cm < 4.0 and vasc < 0.3:
        stage = "Stage II"
    elif vasc < 0.5 and lung_burden < 20:
        stage = "Stage III"
    else:
        stage = "Stage IV"

    return {
        "tumor_config": {
            "sizeCm": round(size_cm, 2),
            "position": [round(pos_x, 4), round(pos_y, 4), round(pos_z, 4)],
            "invasionDepth": round(invasion, 3),
            "necrosisPct": round(necrosis, 3),
            "vascularInvasion": round(vasc, 3),
            "glowIntensity": round(glow, 3),
        },
        "organs": organs,
        "stage_hint": stage,
        "risk_score": risk,
    }
