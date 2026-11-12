"""Folder-based DICOM ingestion and asynchronous segmentation API."""
import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, List
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.app.config import settings
from backend.app.core.auth import User, Permission
from backend.app.core.dependencies import get_optional_user, require_permission
from backend.app.services.imaging_pipeline import PipelineError, run_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/patients/{patient_id}/imaging", tags=["imaging"])

ALLOWED_EXTENSIONS = {".dcm", ".ima", ".dicom"}
_imaging_studies: dict[str, list[dict[str, Any]]] = {}


def _safe_name(filename: str | None) -> str:
    name = Path(filename or "slice.dcm").name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _study(patient_id: str, study_id: str) -> dict[str, Any]:
    study = next((s for s in _imaging_studies.get(patient_id, []) if s["study_id"] == study_id), None)
    if not study:
        raise HTTPException(404, "Imaging study not found")
    return study


def _write_study(study: dict[str, Any]) -> None:
    Path(study["study_dir"]).joinpath("study.json").write_text(json.dumps(study, indent=2), encoding="utf-8")


async def _process(study: dict[str, Any], engine: str) -> None:
    study["status"] = "processing"
    study["workflow"] = ["DICOM validation", "volume reconstruction", f"{engine} segmentation", "surface mesh export"]
    _write_study(study)
    try:
        result = await asyncio.to_thread(run_pipeline, Path(study["study_dir"]), engine)
        study["status"] = "prediction_ready"
        study["prediction"] = result
        study["workflow"].append("prediction ready")
    except PipelineError as exc:
        study["status"] = "failed"
        study["error"] = str(exc)
        logger.warning("Imaging pipeline failed for %s: %s", study["study_id"], exc)
    except Exception:
        study["status"] = "failed"
        study["error"] = "Imaging pipeline failed; inspect server logs."
        logger.exception("Imaging pipeline crashed for %s", study["study_id"])
    finally:
        _write_study(study)


@router.post("/studies")
async def upload_imaging_study(
    patient_id: str,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    engine: str = "monai",
    _user: User | None = Depends(get_optional_user),
):
    """Upload a browser-selected DICOM folder and queue real segmentation."""
    if engine not in {"monai", "nnunet"}:
        raise HTTPException(422, "engine must be 'monai' or 'nnunet'")
    if len(files) < 2:
        raise HTTPException(400, "Upload the complete DICOM series folder (at least two slices)")
    study_id = str(uuid4())
    study_dir = Path(settings.UPLOAD_DIR) / _safe_name(patient_id) / study_id
    study_dir.mkdir(parents=True, exist_ok=False)
    saved: list[dict[str, Any]] = []
    for index, upload in enumerate(files):
        name = _safe_name(upload.filename)
        # Browsers can report an empty MIME type for DICOM. Validate by extension,
        # then pydicom validates the actual dataset inside the worker.
        # Many PACS exports use extensionless DICOM files.  Their bytes are
        # subsequently validated by pydicom during series reconstruction.
        if Path(name).suffix and Path(name).suffix.lower() not in ALLOWED_EXTENSIONS:
            raise HTTPException(415, f"Only DICOM slices may be submitted to this endpoint: {name}")
        target = study_dir / f"{index:05d}_{name}"
        total = 0
        with target.open("wb") as out:
            while chunk := await upload.read(1024 * 1024):
                total += len(chunk)
                if total > 512 * 1024 * 1024:
                    raise HTTPException(413, f"Slice is too large: {name}")
                out.write(chunk)
        saved.append({"filename": name, "file_size": total})

    study: dict[str, Any] = {
        "study_id": study_id, "patient_id": patient_id, "study_dir": str(study_dir),
        "file_count": len(saved), "files": saved, "engine": engine, "status": "queued",
        "workflow": ["upload complete", "DICOM validation", "volume reconstruction", f"{engine} segmentation", "surface mesh export"],
    }
    _imaging_studies.setdefault(patient_id, []).append(study)
    _write_study(study)
    background_tasks.add_task(_process, study, engine)
    return study


@router.post("/studies/{study_id}/process")
async def process_imaging_study(patient_id: str, study_id: str, background_tasks: BackgroundTasks, engine: str = "monai", _user: User | None = Depends(get_optional_user)):
    study = _study(patient_id, study_id)
    if study["status"] not in {"queued", "failed"}:
        return study
    background_tasks.add_task(_process, study, engine)
    return study


@router.get("/studies")
async def get_imaging_studies(patient_id: str, _user: User | None = Depends(get_optional_user)):
    return _imaging_studies.get(patient_id, [])


@router.get("/studies/{study_id}")
async def get_imaging_study(patient_id: str, study_id: str, _user: User | None = Depends(get_optional_user)):
    return _study(patient_id, study_id)


@router.get("/studies/{study_id}/mesh")
async def get_prediction_mesh(patient_id: str, study_id: str, _user: User | None = Depends(get_optional_user)):
    study = _study(patient_id, study_id)
    if study["status"] != "prediction_ready":
        raise HTTPException(409, "Prediction mesh is not ready")
    mesh = Path(study["study_dir"]) / "tumour_mesh.json"
    if not mesh.is_file():
        raise HTTPException(404, "Prediction mesh artifact is missing")
    return FileResponse(mesh, media_type="application/json")
