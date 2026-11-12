"""Workflow state machine router."""
import logging
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends

from backend.app.core.auth import User
from backend.app.core.dependencies import get_current_user, get_optional_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/workflow", tags=["workflow"])

WORKFLOW_STEPS = [
    "registry",
    "clinical",
    "labs",
    "imaging",
    "genomics",
    "validation",
    "analysis",
    "twin",
    "progression",
    "simulation",
    "comparison",
    "report",
    "export",
]

_workflows: dict[str, dict] = {}


@router.post("/init")
async def initialize_workflow(
    data: dict[str, Any],
    user: User | None = Depends(get_optional_user),
):
    patient_id = data.get("patient_id", "")
    wf_id = str(uuid4())
    workflow = {
        "workflow_id": wf_id,
        "patient_id": patient_id,
        "user_id": user.id if user else "",
        "current_step": "registry",
        "completed_steps": [],
        "step_data": {},
        "status": "in_progress",
        "started_at": data.get("started_at"),
        "steps": WORKFLOW_STEPS,
    }
    _workflows[wf_id] = workflow
    return workflow


@router.get("/{patient_id}")
async def get_workflow(
    patient_id: str,
    user: User | None = Depends(get_optional_user),
):
    for wf in _workflows.values():
        if wf["patient_id"] == patient_id:
            completed = set(wf["completed_steps"])
            current = wf["current_step"]
            current_idx = WORKFLOW_STEPS.index(current) if current in WORKFLOW_STEPS else 0
            next_step = WORKFLOW_STEPS[current_idx + 1] if current_idx + 1 < len(WORKFLOW_STEPS) else None
            return {
                "workflow_id": wf["workflow_id"],
                "patient_id": patient_id,
                "current_step": current,
                "completed_steps": wf["completed_steps"],
                "next_step": next_step,
                "progress_pct": round(len(completed) / len(WORKFLOW_STEPS) * 100, 1),
                "is_complete": len(completed) >= len(WORKFLOW_STEPS),
                "status": wf["status"],
            }
    return {
        "patient_id": patient_id,
        "current_step": "registry",
        "completed_steps": [],
        "next_step": "registry",
        "progress_pct": 0,
        "is_complete": False,
        "status": "not_started",
    }


@router.post("/{patient_id}/advance")
async def advance_workflow(
    patient_id: str,
    data: dict[str, Any],
    user: User | None = Depends(get_optional_user),
):
    step = data.get("step", "")
    target_status = data.get("status", "completed")

    wf = None
    for w in _workflows.values():
        if w["patient_id"] == patient_id:
            wf = w
            break

    if not wf:
        wf_id = str(uuid4())
        wf = {
            "workflow_id": wf_id,
            "patient_id": patient_id,
            "user_id": user.id if user else "",
            "current_step": "registry",
            "completed_steps": [],
            "step_data": {},
            "status": "in_progress",
            "steps": WORKFLOW_STEPS,
        }
        _workflows[wf_id] = wf

    if target_status == "completed" and step not in wf["completed_steps"]:
        wf["completed_steps"].append(step)
        current_idx = WORKFLOW_STEPS.index(step) if step in WORKFLOW_STEPS else -1
        if current_idx + 1 < len(WORKFLOW_STEPS):
            wf["current_step"] = WORKFLOW_STEPS[current_idx + 1]

    if len(wf["completed_steps"]) >= len(WORKFLOW_STEPS):
        wf["status"] = "completed"

    return {
        "workflow_id": wf["workflow_id"],
        "patient_id": patient_id,
        "current_step": wf["current_step"],
        "completed_steps": wf["completed_steps"],
        "progress_pct": round(len(wf["completed_steps"]) / len(WORKFLOW_STEPS) * 100, 1),
        "is_complete": wf["status"] == "completed",
        "status": wf["status"],
    }
