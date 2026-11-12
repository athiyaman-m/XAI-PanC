"""Genomics router."""
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException

from backend.app.core.auth import User, Permission
from backend.app.core.dependencies import get_current_user, require_permission

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/patients/{patient_id}/genomics", tags=["genomics"])

_genomic_data: dict[str, dict] = {}


@router.post("")
async def save_genomics(
    patient_id: str,
    data: dict[str, Any],
    _user: User = Depends(require_permission(Permission.WRITE_GENOMICS)),
):
    record = {
        "patient_id": patient_id,
        "kras": data.get("kras", ""),
        "tp53": data.get("tp53", ""),
        "smad4": data.get("smad4", ""),
        "kras_expr": data.get("kras_expr", 0),
        "tp53_expr": data.get("tp53_expr", 0),
        "smad4_expr": data.get("smad4_expr", 0),
        "erbb2_expr": data.get("erbb2_expr", 0),
        "sample_id": data.get("sample_id", f"SMPL-{str(uuid4())[:8]}"),
    }
    _genomic_data[patient_id] = record
    return record


@router.get("")
async def get_genomics(
    patient_id: str,
    _user: User = Depends(require_permission(Permission.READ_GENOMICS)),
):
    return _genomic_data.get(patient_id, {})
