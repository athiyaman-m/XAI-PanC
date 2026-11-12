"""Digital Twin router."""
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.auth import User, Permission
from backend.app.core.dependencies import get_current_user, get_optional_user, require_permission

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/patients/{patient_id}/twin", tags=["digital-twin"])

_twins: dict[str, dict] = {}


@router.post("/generate")
async def generate_twin(
    patient_id: str,
    data: dict[str, Any],
    _user: User | None = Depends(get_optional_user),
):
    twin_id = str(uuid4())
    twin = {
        "twin_id": twin_id,
        "patient_id": patient_id,
        "status": "active",
        "organ_risks": data.get("organ_risks", {}),
        "tumour_volume": data.get("tumour_volume", 0),
        "health_score": data.get("health_score", 100),
        "risk_score": data.get("risk_score", 0),
        "simulation_months": 0,
        "applied_treatment": data.get("applied_treatment"),
        "created_at": data.get("created_at"),
    }
    _twins[twin_id] = twin
    return twin


@router.get("")
async def get_twins(
    patient_id: str,
    _user: User = Depends(require_permission(Permission.READ_TWIN)),
):
    return [t for t in _twins.values() if t["patient_id"] == patient_id]


@router.get("/{twin_id}")
async def get_twin(
    patient_id: str,
    twin_id: str,
    _user: User = Depends(require_permission(Permission.READ_TWIN)),
):
    twin = _twins.get(twin_id)
    if not twin or twin["patient_id"] != patient_id:
        raise HTTPException(404, "Twin not found")
    return twin


@router.post("/{twin_id}/simulate")
async def simulate_twin(
    patient_id: str,
    twin_id: str,
    data: dict[str, Any],
    _user: User | None = Depends(get_optional_user),
):
    twin = _twins.get(twin_id)
    if not twin or twin["patient_id"] != patient_id:
        raise HTTPException(404, "Twin not found")

    treatment = data.get("treatment", "Best Supportive Care")
    months = data.get("months", 3)
    tumour_volume = data.get("tumour_volume_cm3", twin.get("tumour_volume", 10))

    effect_map = {
        "FOLFIRINOX": 0.78,
        "Gemcitabine + nab-paclitaxel": 0.86,
        "Radiotherapy": 0.90,
        "Targeted therapy": 0.83,
        "Immunotherapy": 0.93,
        "Whipple surgery": 0.25,
        "Best Supportive Care": 0.96,
    }
    effect = effect_map.get(treatment, 0.96)
    new_volume = round(tumour_volume * (1 + 0.13 * months) * effect, 2)
    response_pct = round(max(0, (1 - new_volume / max(tumour_volume, 0.1)) * 100), 1)
    ca199_relative = round(new_volume / max(tumour_volume, 0.1) * 0.92, 2)

    twin["tumour_volume"] = new_volume
    twin["simulation_months"] = months
    twin["applied_treatment"] = treatment

    return {
        "month": months,
        "treatment": treatment,
        "tumour_volume_cm3": new_volume,
        "response_pct": response_pct,
        "ca199_relative": ca199_relative,
        "organ_risk_update": "Twin updated",
        "disclaimer": "Research simulation only. Not a validated clinical recommendation.",
    }
