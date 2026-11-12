"""Patient CRUD router."""
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.app.core.auth import User, UserRole, Permission
from backend.app.core.dependencies import get_current_user, require_permission

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/patients", tags=["patients"])

# In-memory patient store for development
_patients: dict[str, dict] = {}
_patients_auto_id = 0


@router.get("")
async def list_patients(
    search: str = Query("", description="Search by name or MRN"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    _user: User = Depends(get_current_user),
):
    result = []
    for pid, p in _patients.items():
        if search and search.lower() not in p.get("name", "").lower() and search.lower() not in p.get("mrn", "").lower():
            continue
        result.append(p)
    return {"items": result[skip : skip + limit], "total": len(result), "skip": skip, "limit": limit}


@router.post("")
async def create_patient(
    data: dict[str, Any],
    _user: User = Depends(require_permission(Permission.WRITE_PATIENT)),
):
    global _patients_auto_id
    _patients_auto_id += 1
    patient_id = data.get("patient_id", f"P{_patients_auto_id:06d}")
    record = {
        "patient_id": patient_id,
        "mrn": data.get("mrn", f"MRN{_patients_auto_id:06d}"),
        "name": data.get("name", ""),
        "date_of_birth": data.get("date_of_birth"),
        "sex": data.get("sex", ""),
        "age": data.get("age", 0),
        "symptoms": data.get("symptoms", ""),
        "ecog": data.get("ecog", ""),
        "diagnosis": data.get("diagnosis", "Pancreatic ductal adenocarcinoma"),
        "user_id": _user.id,
        "is_active": True,
    }
    _patients[patient_id] = record
    logger.info("Created patient %s", patient_id)
    return record


@router.get("/{patient_id}")
async def get_patient(
    patient_id: str,
    _user: User = Depends(get_current_user),
):
    patient = _patients.get(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.put("/{patient_id}")
async def update_patient(
    patient_id: str,
    data: dict[str, Any],
    _user: User = Depends(require_permission(Permission.WRITE_PATIENT)),
):
    patient = _patients.get(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    patient.update(data)
    logger.info("Updated patient %s", patient_id)
    return patient


@router.delete("/{patient_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_patient(
    patient_id: str,
    _user: User = Depends(require_permission(Permission.DELETE_PATIENT)),
):
    if patient_id not in _patients:
        raise HTTPException(status_code=404, detail="Patient not found")
    del _patients[patient_id]


@router.get("/{patient_id}/clinical")
async def get_clinical_data(
    patient_id: str,
    _user: User = Depends(get_current_user),
):
    if patient_id not in _patients:
        raise HTTPException(status_code=404, detail="Patient not found")
    return {
        "patient_id": patient_id,
        "ca199": _patients[patient_id].get("ca199", 0),
        "cea": _patients[patient_id].get("cea", 0),
        "bilirubin": _patients[patient_id].get("bilirubin", 0),
        "albumin": _patients[patient_id].get("albumin", 0),
        "ecog": _patients[patient_id].get("ecog", ""),
        "symptoms": _patients[patient_id].get("symptoms", ""),
    }
