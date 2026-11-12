"""Clinical assessment router."""
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.auth import User, Permission
from backend.app.core.dependencies import get_current_user, require_permission

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/patients/{patient_id}/clinical", tags=["clinical"])

_clinical_assessments: dict[str, list] = {}


@router.post("")
async def create_clinical_assessment(
    patient_id: str,
    data: dict[str, Any],
    _user: User = Depends(require_permission(Permission.WRITE_CLINICAL)),
):
    if patient_id not in _clinical_assessments:
        _clinical_assessments[patient_id] = []
    assessment = {
        "assessment_id": str(uuid4()),
        "patient_id": patient_id,
        "symptoms": data.get("symptoms", ""),
        "ecog": data.get("ecog", ""),
        "diagnosis": data.get("diagnosis", "Pancreatic ductal adenocarcinoma"),
        "physical_exam": data.get("physical_exam", ""),
        "treatment_plan": data.get("treatment_plan", ""),
        "notes": data.get("notes", ""),
        "recorded_at": data.get("recorded_at"),
    }
    _clinical_assessments[patient_id].append(assessment)
    return assessment


@router.get("")
async def get_clinical_assessments(
    patient_id: str,
    _user: User = Depends(require_permission(Permission.READ_CLINICAL)),
):
    return _clinical_assessments.get(patient_id, [])
