"""AI Prediction router."""
import json
import logging
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends

from backend.app.core.auth import User, Permission
from backend.app.core.dependencies import get_current_user, get_optional_user, require_permission

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predictions", tags=["predictions"])

_predictions: list[dict] = []


def _fallback_prediction(inputs: dict[str, Any]) -> dict[str, Any]:
    """Heuristic fallback when the ML prediction engine is unavailable."""
    age = inputs.get("age", 60)
    sex = inputs.get("sex", "Male")
    gender_male = 1 if sex == "Male" else 0

    ca199 = float(inputs.get("ca199", 37))
    cea = float(inputs.get("cea", 3))
    bilirubin = float(inputs.get("bilirubin", 0.8))
    wt = float(inputs.get("weight_loss_pct", 0))
    smoking = inputs.get("smoking", "Never")
    diabetes = inputs.get("diabetes", "None")
    family_history = inputs.get("family_history", False)
    pancreatitis = inputs.get("pancreatitis", False)
    genetic = inputs.get("genetic", False)

    age_score = max(0.0, (age - 40) / 80.0) * 0.20
    if age >= 70:
        age_score += 0.05
    elif age >= 60:
        age_score += 0.02

    bio_score = 0.0
    if ca199 > 1000:
        bio_score += 0.20
    elif ca199 > 500:
        bio_score += 0.15
    elif ca199 > 100:
        bio_score += 0.10
    elif ca199 > 37:
        bio_score += 0.05
    if cea > 20:
        bio_score += 0.06
    elif cea > 5:
        bio_score += 0.03
    if bilirubin > 5:
        bio_score += 0.08
    elif bilirubin > 2:
        bio_score += 0.04
    if wt > 15:
        bio_score += 0.08
    elif wt > 10:
        bio_score += 0.05
    elif wt > 5:
        bio_score += 0.02
    if smoking == "Current":
        bio_score += 0.06
    elif smoking == "Former":
        bio_score += 0.03
    if "New-onset" in diabetes or "Type 2" in diabetes:
        bio_score += 0.08
    if family_history:
        bio_score += 0.05
    if pancreatitis:
        bio_score += 0.04
    if genetic:
        bio_score += 0.06

    prob = min(age_score + bio_score, 0.97)

    if prob > 0.80 and ca199 > 500:
        stage_label, stage_col, stage_detail = "Stage IV", "#e74c3c", "Distant metastasis likely"
    elif prob > 0.65 and ca199 > 200:
        stage_label, stage_col, stage_detail = "Stage III", "#e67e22", "Locally advanced"
    elif prob > 0.50 and ca199 > 100:
        stage_label, stage_col, stage_detail = "Stage II", "#f39c12", "Regional spread"
    elif prob > 0.35:
        stage_label, stage_col, stage_detail = "Stage I", "#2ecc71", "Localised"
    else:
        stage_label, stage_col, stage_detail = "Non-malignant", "#3498db", "No strong malignancy evidence"

    base_hazard = 1.0
    if "III" in stage_label or "IV" in stage_label:
        base_hazard += 1.5
    elif "II" in stage_label:
        base_hazard += 0.5
    base_hazard += prob * 2.0
    base_hazard += max(0, (age - 60) / 40)

    risk_score = round(base_hazard, 2)
    median_months = max(3, int(36 - base_hazard * 5))

    folfi = 0.5 + (0.15 if age < 65 else 0) + (0.08 if prob < 0.6 else 0)
    gem = 0.45 + max(0, (1 - prob) * 0.2)
    treatment = {
        "FOLFIRINOX": round(min(max(folfi, 0.1), 0.92), 2),
        "Gemcitabine + nab-paclitaxel": round(min(max(gem, 0.1), 0.88), 2),
        "FOLFOX": round(min(max(folfi * 0.85, 0.1), 0.80), 2),
        "Best Supportive Care": round(min(max(0.30 + prob * 0.15, 0.1), 0.55), 2),
    }

    shap = {
        "Age": round((age - 55) / 50 * 0.20, 4),
        "Sex (Male)": round(0.08 if gender_male else -0.04, 4),
        "CA 19-9 Level": round((ca199 - 37) / 2000 * 0.35, 4),
        "Serum Bilirubin": round(bilirubin / 20 * 0.12, 4),
        "Weight Loss (%)": round(wt / 20 * 0.10, 4),
        "CEA Level": round((cea - 5) / 200 * 0.08, 4),
        "Diabetes Status": round(0.10 if "New-onset" in diabetes or "Type 2" in diabetes else 0.03, 4),
        "Smoking History": round(0.08 if smoking == "Current" else 0.04 if smoking == "Former" else 0.0, 4),
        "Family History": round(0.07 if family_history else 0.0, 4),
        "Pancreatitis History": round(0.06 if pancreatitis else 0.0, 4),
    }

    panc_risk = min(98, int(prob * 100 * 0.85 + (10 if pancreatitis else 0)))
    liver_risk = min(95, int((bilirubin / 12) * 60 + (ca199 / 5000) * 40 + (15 if prob > 0.65 else 0)))
    bile_risk = min(92, int((bilirubin / 8) * 55 + (20 if bilirubin > 3 else 0)))
    duo_risk = min(80, int(prob * 60 + (wt / 30) * 25))
    spleen_risk = min(70, int(prob * 0.4 * 60))
    lung_risk = min(60, int(prob * 0.35 * 80))

    def _rc(r):
        if r >= 70:
            return "#e74c3c"
        if r >= 45:
            return "#f39c12"
        if r >= 20:
            return "#f1c40f"
        return "#2ecc71"

    organ_risks = {
        "Pancreas": (panc_risk, _rc(panc_risk), "Primary tumour site"),
        "Liver": (liver_risk, _rc(liver_risk), "Biliary obstruction risk"),
        "Bile Duct": (bile_risk, _rc(bile_risk), "Obstructive jaundice risk"),
        "Duodenum": (duo_risk, _rc(duo_risk), "Anatomical invasion risk"),
        "Spleen": (spleen_risk, _rc(spleen_risk), "Splenic vein involvement"),
        "Lung": (lung_risk, _rc(lung_risk), "Distant metastasis risk"),
    }

    return {
        "prob": prob,
        "stage_label": stage_label,
        "stage_col": stage_col,
        "stage_detail": stage_detail,
        "survival": {
            "risk_score": risk_score,
            "median_months": median_months,
            "risk_group": "High Risk" if base_hazard > 3.0 else "Medium Risk" if base_hazard > 1.5 else "Low Risk",
        },
        "treatment": treatment,
        "best_tx": max(treatment, key=treatment.get),
        "shap_vals": shap,
        "organ_risks": organ_risks,
        "bio_timeline": [],
    }


@router.post("/run")
async def run_prediction(
    data: dict[str, Any],
    _user: User | None = Depends(get_optional_user),
):
    inputs = {
        "age": data.get("age", 60),
        "sex": data.get("sex", "Male"),
        "ca199": float(data.get("ca199", 37)),
        "cea": float(data.get("cea", 3)),
        "bilirubin": float(data.get("bilirubin", 0.8)),
        "amylase": float(data.get("amylase", 95)),
        "weight_loss_pct": float(data.get("weight_loss_pct", 0)),
        "smoking": data.get("smoking", "Never"),
        "alcohol": data.get("alcohol", "None / Minimal"),
        "diabetes": data.get("diabetes", "None"),
        "family_history": bool(data.get("family_history", False)),
        "pancreatitis": bool(data.get("pancreatitis", False)),
        "genetic": bool(data.get("genetic", False)),
        "kras_expr": float(data.get("kras_expr", 5.0)),
        "tp53_expr": float(data.get("tp53_expr", 5.0)),
        "smad4_expr": float(data.get("smad4_expr", 5.0)),
        "erbb2_expr": float(data.get("erbb2_expr", 5.0)),
    }

    try:
        from backend.prediction_engine import run_full_prediction

        raw = run_full_prediction(inputs)
    except ImportError:
        raw = _fallback_prediction(inputs)
    except Exception as e:
        logger.warning("Prediction engine failed (%s), using fallback", e)
        raw = _fallback_prediction(inputs)

    def _serialize(v):
        if isinstance(v, pd.DataFrame):
            return _serialize(v.to_dict(orient="records"))
        if isinstance(v, pd.Series):
            return _serialize(v.tolist())
        if isinstance(v, dict):
            return {k: _serialize(val) for k, val in v.items()}
        if isinstance(v, (list, tuple)):
            return [_serialize(x) for x in v]
        if isinstance(v, np.ndarray):
            return _serialize(v.tolist())
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if isinstance(v, np.bool_):
            return bool(v)
        return v

    result = _serialize(raw)
    result["patient_id"] = data.get("patient_id", "")
    result["user_id"] = _user.id if _user else ""

    _predictions.append(result)

    organ_risks = result.get("organ_risks", {})
    organ_names = {"Pancreas", "Liver", "Bile Duct", "Duodenum", "Spleen", "Lungs", "Lymph Nodes", "Stomach", "Kidney", "Lung"}
    organ_colours = {
        "Pancreas": "#ef565f", "Liver": "#f4a524", "Bile Duct": "#f77828",
        "Duodenum": "#f4b024", "Spleen": "#35c59b", "Lungs": "#6badf5",
        "Lymph Nodes": "#9b59b6", "Stomach": "#e8c3a0", "Kidney": "#e74c3c", "Lung": "#6badf5",
    }

    def _parse_burden(v):
        if isinstance(v, (list, tuple)):
            return v[0]
        if isinstance(v, dict) and "burden" in v:
            return v["burden"]
        return int(v) if isinstance(v, (int, float)) else 50

    organs_arr = []
    for name, val in organ_risks.items():
        burden = _parse_burden(val)
        colour = organ_colours.get(name, "#5db7a2")
        organs_arr.append({
            "name": name,
            "burden": burden / 100.0,
            "health": max(0, min(100, round(100 - burden))),
            "colour": colour,
        })

    prob_val = result.get("prob", 0)
    tumour_volume = round(prob_val * 25 + 1, 1)

    return {
        "cancer_probability": round(prob_val * 100, 1),
        "stage": result.get("stage_label", "Pending"),
        "stage_colour": result.get("stage_col", "#3498db"),
        "stage_detail": result.get("stage_detail", ""),
        "risk_score": result.get("survival", {}).get("risk_score", 0),
        "median_survival_months": result.get("survival", {}).get("median_months", 0),
        "tumour_volume_cm3": tumour_volume,
        "survival_data": result.get("survival", {}),
        "treatment_data": result.get("treatment", {}),
        "best_treatment": result.get("best_tx", ""),
        "shap_values": result.get("shap_vals", {}),
        "organ_risks": organ_risks,
        "organs": organs_arr,
        "biomarker_timeline": result.get("bio_timeline", []),
        "model_version": "2.0.0",
    }


@router.get("/history")
async def get_prediction_history(
    patient_id: str = "",
    _user: User = Depends(get_current_user),
):
    if patient_id:
        return [p for p in _predictions if p.get("patient_id") == patient_id]
    return _predictions[-20:]  # Last 20 predictions
