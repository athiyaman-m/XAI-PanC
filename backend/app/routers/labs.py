"""Lab results router."""
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends

from backend.app.core.auth import User, Permission
from backend.app.core.dependencies import get_current_user, require_permission

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/patients/{patient_id}/labs", tags=["laboratory"])

_lab_results: dict[str, list] = {}


@router.post("")
async def create_lab_result(
    patient_id: str,
    data: dict[str, Any],
    _user: User = Depends(require_permission(Permission.WRITE_CLINICAL)),
):
    if patient_id not in _lab_results:
        _lab_results[patient_id] = []
    result = {
        "result_id": str(uuid4()),
        "patient_id": patient_id,
        "test_name": data.get("test_name", ""),
        "value": data.get("value", 0),
        "unit": data.get("unit", ""),
        "reference_range": data.get("reference_range", ""),
        "collected_at": data.get("collected_at"),
    }
    _lab_results[patient_id].append(result)
    return result


@router.post("/batch")
async def create_lab_results_batch(
    patient_id: str,
    data: dict[str, Any],
    _user: User = Depends(require_permission(Permission.WRITE_CLINICAL)),
):
    if patient_id not in _lab_results:
        _lab_results[patient_id] = []
    results = []
    for test_data in data.get("tests", []):
        result = {
            "result_id": str(uuid4()),
            "patient_id": patient_id,
            "test_name": test_data.get("test_name", ""),
            "value": test_data.get("value", 0),
            "unit": test_data.get("unit", ""),
            "reference_range": test_data.get("reference_range", ""),
            "collected_at": test_data.get("collected_at"),
        }
        _lab_results[patient_id].append(result)
        results.append(result)
    return {"results": results, "count": len(results)}


@router.get("")
async def get_lab_results(
    patient_id: str,
    _user: User = Depends(require_permission(Permission.READ_CLINICAL)),
):
    return _lab_results.get(patient_id, [])
