"""Treatment recommendation and comparison router."""
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends

from backend.app.core.auth import User, Permission
from backend.app.core.dependencies import get_current_user, get_optional_user, require_permission

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/treatment", tags=["treatment"])

_treatment_records: list[dict] = []

TREATMENTS = {
    "FOLFIRINOX": {
        "name": "FOLFIRINOX",
        "category": "Chemotherapy",
        "efficacy_score": 85,
        "toxicity_score": 72,
        "patient_suitability": "Younger, fit patients (ECOG 0-1)",
        "median_os_months": 11.1,
        "response_rate_pct": 31.6,
        "common_side_effects": ["Myelosuppression", "Diarrhea", "Fatigue", "Nausea"],
    },
    "Gemcitabine + nab-paclitaxel": {
        "name": "Gemcitabine + nab-paclitaxel",
        "category": "Chemotherapy",
        "efficacy_score": 78,
        "toxicity_score": 65,
        "patient_suitability": "Elderly or comorbid patients",
        "median_os_months": 8.5,
        "response_rate_pct": 23.0,
        "common_side_effects": ["Neuropathy", "Fatigue", "Myelosuppression"],
    },
    "FOLFOX": {
        "name": "FOLFOX",
        "category": "Chemotherapy",
        "efficacy_score": 72,
        "toxicity_score": 60,
        "patient_suitability": "Alternative for fluoropyrimidine-sensitive patients",
        "median_os_months": 6.8,
        "response_rate_pct": 17.0,
        "common_side_effects": ["Neuropathy", "Myelosuppression", "Fatigue"],
    },
    "Radiotherapy": {
        "name": "Radiotherapy",
        "category": "Radiation",
        "efficacy_score": 65,
        "toxicity_score": 45,
        "patient_suitability": "Locally advanced, unresectable",
        "response_rate_pct": 15.0,
        "common_side_effects": ["Fatigue", "Skin changes", "Nausea"],
    },
    "Targeted therapy": {
        "name": "Targeted Therapy",
        "category": "Targeted",
        "efficacy_score": 70,
        "toxicity_score": 50,
        "patient_suitability": "Patients with targetable mutations",
        "response_rate_pct": 20.0,
        "common_side_effects": ["Rash", "Diarrhea", "Fatigue"],
    },
    "Immunotherapy": {
        "name": "Immunotherapy",
        "category": "Immunotherapy",
        "efficacy_score": 55,
        "toxicity_score": 55,
        "patient_suitability": "MSI-H or dMMR tumors",
        "response_rate_pct": 12.0,
        "common_side_effects": ["Immune-related adverse events", "Fatigue", "Rash"],
    },
    "Whipple surgery": {
        "name": "Whipple Surgery (Pancreaticoduodenectomy)",
        "category": "Surgery",
        "efficacy_score": 90,
        "toxicity_score": 80,
        "patient_suitability": "Resectable disease, good performance status",
        "median_os_months": 24.0,
        "response_rate_pct": 80.0,
        "common_side_effects": ["Delayed gastric emptying", "Infection", "Pancreatic leak"],
    },
    "Best Supportive Care": {
        "name": "Best Supportive Care",
        "category": "Palliative",
        "efficacy_score": 30,
        "toxicity_score": 10,
        "patient_suitability": "Poor performance status, extensive disease",
        "median_os_months": 3.0,
        "common_side_effects": [],
    },
}


@router.get("/options")
async def get_treatment_options(
    _user: User | None = Depends(get_optional_user),
):
    return {"treatments": list(TREATMENTS.values())}


@router.post("/recommend")
async def recommend_treatments(
    data: dict[str, Any],
    _user: User | None = Depends(get_optional_user),
):
    age = data.get("age", 60)
    ecog = data.get("ecog", 0)
    stage = data.get("stage", "Stage III")
    is_late = "III" in stage or "IV" in stage

    scored = []
    for key, tx in TREATMENTS.items():
        score = tx["efficacy_score"]
        if age > 70 and tx["toxicity_score"] > 70:
            score -= 15
        if ecog > 1 and tx["toxicity_score"] > 60:
            score -= 10
        if is_late and tx["category"] in ("Surgery",):
            score -= 30
        if not is_late and tx["category"] in ("Surgery",):
            score += 10
        scored.append({**tx, "adjusted_score": max(0, min(100, score))})

    scored.sort(key=lambda x: x["adjusted_score"], reverse=True)
    return {"recommendations": scored, "best_treatment": scored[0]["name"] if scored else None}


@router.post("/compare")
async def compare_treatments(
    data: dict[str, Any],
    _user: User | None = Depends(get_optional_user),
):
    selected = data.get("treatments", ["FOLFIRINOX", "Gemcitabine + nab-paclitaxel"])
    age = data.get("age", 60)
    stage = data.get("stage", "Stage III")
    is_late = "III" in stage or "IV" in stage

    comparison = []
    for key in selected:
        if key in TREATMENTS:
            tx = dict(TREATMENTS[key])
            est_response = tx.get("response_rate_pct", 20)
            if age > 70 and tx.get("toxicity_score", 50) > 70:
                est_response *= 0.7
            if is_late and tx["category"] == "Surgery":
                est_response *= 0.3
            tx["estimated_response_pct"] = round(est_response, 1)
            comparison.append(tx)

    return {
        "comparison_id": str(uuid4()),
        "treatments": comparison,
        "age": age,
        "stage": stage,
        "best_match": max(comparison, key=lambda x: x.get("estimated_response_pct", 0))["name"] if comparison else None,
    }


@router.post("/records")
async def save_treatment_record(
    data: dict[str, Any],
    _user: User = Depends(require_permission(Permission.WRITE_CLINICAL)),
):
    record = {
        "treatment_id": str(uuid4()),
        "patient_id": data.get("patient_id"),
        "treatment_name": data.get("treatment_name"),
        "status": data.get("status", "recommended"),
        "start_date": data.get("start_date"),
        "end_date": data.get("end_date"),
        "dosage": data.get("dosage"),
        "notes": data.get("notes"),
    }
    _treatment_records.append(record)
    return record
