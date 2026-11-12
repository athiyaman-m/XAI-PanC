"""Disease progression and treatment simulation router."""
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends

from backend.app.core.auth import User, Permission
from backend.app.core.dependencies import get_current_user, get_optional_user, require_permission

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/simulation", tags=["simulation"])

_simulations: list[dict] = []


@router.post("/run")
async def run_simulation(
    data: dict[str, Any],
    _user: User | None = Depends(get_optional_user),
):
    treatment = data.get("treatment", "FOLFIRINOX")
    months = data.get("months", 6)
    initial_volume = data.get("initial_volume_cm3", 12.4)
    ca199 = data.get("ca199", 824)
    age = data.get("age", 60)

    effect_map = {
        "FOLFIRINOX": 0.78,
        "Gemcitabine + nab-paclitaxel": 0.86,
        "FOLFOX": 0.85,
        "Radiotherapy": 0.90,
        "Targeted therapy": 0.83,
        "Immunotherapy": 0.93,
        "Whipple surgery": 0.25,
        "Best Supportive Care": 0.96,
    }
    effect = effect_map.get(treatment, 0.96)

    simulation_id = str(uuid4())
    timeline = []
    for m in range(months + 1):
        vol = round(initial_volume * (1 + 0.13 * m) * effect, 2)
        ca = round(ca199 * (1 + 0.05 * m) * effect, 1)
        timeline.append({"month": m, "tumour_volume_cm3": vol, "ca199": ca})

    final_volume = timeline[-1]["tumour_volume_cm3"]
    response_pct = round(max(0, (1 - final_volume / max(initial_volume, 0.1)) * 100), 1)

    result = {
        "simulation_id": simulation_id,
        "treatment": treatment,
        "months": months,
        "initial_volume_cm3": initial_volume,
        "final_volume_cm3": final_volume,
        "response_pct": response_pct,
        "timeline": timeline,
        "status": "completed",
    }
    _simulations.append(result)
    return result


@router.get("/history")
async def get_simulation_history(
    _user: User = Depends(get_current_user),
):
    return _simulations[-20:]
