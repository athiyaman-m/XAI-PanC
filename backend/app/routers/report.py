"""Clinical report generation and export router."""
import json
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends

from backend.app.core.auth import User, Permission
from backend.app.core.dependencies import get_current_user, require_permission

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/patients/{patient_id}/reports", tags=["reports"])

_reports: dict[str, list] = {}


@router.post("/generate")
async def generate_report(
    patient_id: str,
    data: dict[str, Any],
    _user: User = Depends(require_permission(Permission.WRITE_REPORTS)),
):
    report_id = str(uuid4())
    report = {
        "report_id": report_id,
        "patient_id": patient_id,
        "report_type": data.get("report_type", "clinical_summary"),
        "format": data.get("format", "json"),
        "content": data.get("content", {}),
        "generated_at": data.get("generated_at"),
        "status": "completed",
    }
    if patient_id not in _reports:
        _reports[patient_id] = []
    _reports[patient_id].append(report)
    return report


@router.get("")
async def get_reports(
    patient_id: str,
    _user: User = Depends(require_permission(Permission.READ_REPORTS)),
):
    return _reports.get(patient_id, [])


@router.get("/latest")
async def get_latest_report(
    patient_id: str,
    _user: User = Depends(require_permission(Permission.READ_REPORTS)),
):
    reports = _reports.get(patient_id, [])
    if not reports:
        return None
    return reports[-1]


@router.post("/export")
async def export_report(
    patient_id: str,
    data: dict[str, Any],
    _user: User = Depends(require_permission(Permission.WRITE_REPORTS)),
):
    report_data = data.get("report_data", {})
    export_format = data.get("format", "pdf")
    export_id = str(uuid4())
    return {
        "export_id": export_id,
        "patient_id": patient_id,
        "format": export_format,
        "status": "queued",
        "download_url": f"/api/v1/patients/{patient_id}/reports/exports/{export_id}/download",
        "message": f"Report export queued in {export_format} format.",
    }
