"""Digital Twin 3D mesh/volume endpoints — Real CT Twin + Realistic Anatomy +
Patient CT (DICOM volume render) + VTK CT Twin (pure VTK pipeline).

Serves patient-specific meshes as JSON in
the ``{positions, indices}`` schema consumed by the React/R3F frontend, plus a
downsampled CT volume payload for the browser volume renderer.

The VTK CT Twin endpoints (``/vtk-ct``, ``/vtk-scenario``) use a pure VTK
pipeline (vtkNIFTIImageReader + vtkMarchingCubes) to extract ground-truth
pancreas and tumor surfaces from NIfTI segmentation masks.
"""
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, Body

from backend.app.config import settings
from backend.app.core.auth import User
from backend.app.core.dependencies import get_optional_user
from backend.app.services.ct_volume_export import (
    anchor_from_mask, build_ct_volume_payload,
)
from backend.app.services.imaging_pipeline import (
    PipelineError, load_dicom_series,
)
from backend.app.services.biomarker_tumor_mapper import map_biomarkers_to_tumor
from backend.app.services.twin_mesh_exporter import (
    available_msd_cases, real_ct_payload, realistic_anatomy_payload,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/twin/mesh", tags=["digital-twin-3d"])

DEFAULT_CASE = "409"
_CT_EXTENSIONS = {".dcm", ".ima", ".dicom"}


def _safe_name(filename: str | None) -> str:
    name = Path(filename or "slice.dcm").name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def _patient_segmentation(patient_id: str, vol_shape) -> tuple[list[int] | None, float | None]:
    """Best-effort tumour anchor from an earlier uploaded DICOM segmentation.

    Returns ``(full_res_voxel, volume_cm3)`` or ``(None, None)`` when no
    segmentation that matches the uploaded series exists.  The caller's
    downsampling strides are applied to the anchor afterwards.
    """
    if not patient_id:
        return None, None
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", patient_id)
    base = Path(settings.UPLOAD_DIR) / safe
    if not base.is_dir():
        return None, None
    segs = sorted(base.rglob("segmentation.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    for seg in segs:
        try:
            mask = np.load(seg)
        except Exception:
            continue
        if tuple(mask.shape) != tuple(vol_shape):
            continue
        voxel = anchor_from_mask(mask, (1, 1, 1))
        cm3 = None
        try:
            study = json.loads((seg.parent / "study.json").read_text(encoding="utf-8"))
            pred = study.get("prediction") or {}
            cm3 = pred.get("tumour_volume_cm3")
            if not cm3:
                ss = pred.get("spacing_mm") or []
                ss = [float(v) for v in ss]
                if ss and np.prod(ss) > 0:
                    cm3 = round(float(np.asarray(mask > 0).sum()) * float(np.prod(ss)) / 1000.0, 3)
        except Exception:
            cm3 = None
        return voxel, cm3
    return None, None


@router.get("/options")
async def twin_mesh_options(_user: User | None = Depends(get_optional_user)):
    cases = available_msd_cases()
    # Also discover cases available for the pure VTK pipeline
    try:
        from digital_twin.vtk_engine import available_vtk_cases
        vtk_cases = available_vtk_cases()
    except Exception:
        vtk_cases = []
    return {
        "modes": ["realistic", "real-ct", "vtk-ct"],
        "default_case": DEFAULT_CASE if DEFAULT_CASE in cases else (cases[-1] if cases else DEFAULT_CASE),
        "ct_cases": cases or [DEFAULT_CASE],
        "vtk_cases": vtk_cases or cases or [DEFAULT_CASE],
    }


@router.get("/realistic")
async def get_realistic_anatomy(_user: User | None = Depends(get_optional_user)):
    try:
        return realistic_anatomy_payload()
    except Exception as exc:
        logger.exception("Realistic anatomy mesh export failed")
        raise HTTPException(500, f"Realistic anatomy mesh export failed: {exc}")


@router.get("/real-ct")
async def get_real_ct_twin(
    case: str = Query(DEFAULT_CASE),
    _user: User | None = Depends(get_optional_user),
):
    try:
        return real_ct_payload(case)
    except ValueError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        logger.exception("Real CT twin mesh export failed for case %s", case)
        raise HTTPException(500, f"Real CT twin mesh export failed: {exc}")


@router.get("/ct-volume")
async def get_ct_volume(
    case: str = Query(DEFAULT_CASE),
    _user: User | None = Depends(get_optional_user),
):
    """Downsampled CT volume for one MSD case (real patient CT), with the
    ground-truth tumour anchor, for the browser volume renderer."""
    from digital_twin.anatomy.real_ct_twin import get_case_meshes, tumor_volume_cm3
    try:
        data = get_case_meshes(case)
        if not data or "vol_crop" not in data:
            raise HTTPException(404, f"MSD case {case} not found or CT volume unavailable")
        vol = np.asarray(data["vol_crop"], dtype=np.float32)
        spacing = tuple(float(v) for v in np.asarray(data["spacing"], dtype=np.float64))
        offset_mm = np.asarray(data.get("offset_mm"), dtype=np.float64)
        tum = np.asarray(data.get("tum_verts"), dtype=np.float64)
        tumor_voxel = None
        if tum is not None and len(tum) and offset_mm is not None and len(offset_mm):
            local_mm = tum - offset_mm
            full = np.clip(np.round(local_mm.mean(axis=0) / spacing), 0, np.array(vol.shape) - 1)
            tumor_voxel = [int(v) for v in full]
        return build_ct_volume_payload(
            vol, spacing,
            title=f"Patient CT — MSD case {case}",
            source="MSD Task07 Pancreas (real CT)",
            tumor_voxel=tumor_voxel,
            tumor_cm3=tumor_volume_cm3(case),
            tumor_source="msd" if tumor_voxel else "default",
            tumor_label="Ground-truth tumour",
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("CT volume export failed for case %s", case)
        raise HTTPException(500, f"CT volume export failed: {exc}")


@router.post("/ct-volume")
async def upload_ct_volume(
    files: list[UploadFile] = File(...),
    patient_id: str | None = Form(default=None),
    _user: User | None = Depends(get_optional_user),
):
    """Reconstruct an uploaded DICOM series and return the volume payload.

    The tumour anchor is taken from the patient's most recent imaging-study
    segmentation when available (used to highlight the real lesion); otherwise
    the renderer falls back to a biomarker-anchored synthetic tumour.
    """
    if len(files) < 2:
        raise HTTPException(400, "Upload the complete DICOM series folder (at least two slices)")
    tmp = Path(tempfile.mkdtemp(prefix="ct_volume_"))
    try:
        for index, upload in enumerate(files):
            name = _safe_name(upload.filename)
            if Path(name).suffix and Path(name).suffix.lower() not in _CT_EXTENSIONS:
                raise HTTPException(415, f"Only DICOM slices may be submitted here: {name}")
            target = tmp / f"{index:05d}_{name}"
            with target.open("wb") as out:
                while chunk := await upload.read(1024 * 1024):
                    out.write(chunk)
        try:
            volume = load_dicom_series(tmp)
        except PipelineError as exc:
            raise HTTPException(400, str(exc))
        voxel, cm3 = _patient_segmentation(patient_id, volume.data.shape)
        source_label = volume.label or "Uploaded DICOM series"
        return build_ct_volume_payload(
            volume.data,
            tuple(float(v) for v in volume.spacing),
            title="Patient CT — uploaded DICOM",
            source=source_label,
            tumor_voxel=voxel,
            tumor_cm3=cm3,
            tumor_source="segmentation" if voxel else "default",
            tumor_label="Detected lesion (segmentation)" if voxel else None,
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@router.post("/biomarker-tumor")
async def biomarker_tumor(
    data: dict,
    _user: User | None = Depends(get_optional_user),
):
    """Instant biomarker → tumor geometry mapping for the reactive 3D twin.

    Accepts biomarker values and returns a ``TumorConfig`` dict plus per-organ
    disease burden percentages.  The mapping is deterministic and < 1 ms, so
    every slider change on the frontend triggers a fresh call.
    """
    return map_biomarkers_to_tumor(data)


# ---------------------------------------------------------------------------
# VTK CT Twin endpoints — pure VTK pipeline
# ---------------------------------------------------------------------------

@router.get("/vtk-ct")
async def get_vtk_ct_twin(
    case: str = Query(DEFAULT_CASE),
    _user: User | None = Depends(get_optional_user),
):
    """Ground-truth pancreas + tumor meshes extracted via VTK Marching Cubes
    from NIfTI segmentation masks.

    The pancreas (label=1) and tumor (label=2) surfaces share the SAME
    physical coordinate system (spacing, origin, orientation) from the
    NIfTI header. Both meshes are returned as ``{positions, indices}``
    arrays in millimetres.

    The response includes full VTK diagnostics (bounds, center, dimensions,
    volume, spacing, origin) for both meshes to verify spatial alignment.
    """
    try:
        from digital_twin.vtk_engine import process_case
        return process_case(case)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        logger.exception("VTK CT twin failed for case %s", case)
        raise HTTPException(500, f"VTK CT twin failed: {exc}")


@router.post("/vtk-scenario")
async def vtk_scenario_tumor(
    data: dict = Body(...),
    _user: User | None = Depends(get_optional_user),
):
    """Generate a simulated/projected tumor as a separate scenario layer.

    The ground-truth tumor geometry is deep-copied and transformed (scaled
    around its centroid + optional translation). The original ground-truth
    mesh is NEVER modified.

    Request body::

        {
            "case": "409",
            "scale": 1.5,
            "shift_mm": [0, 0, 0]
        }

    Returns a single organ entry with ``layer='scenario'`` plus diagnostics.
    """
    try:
        from digital_twin.vtk_engine import generate_scenario_tumor
        case_id = data.get("case", DEFAULT_CASE)
        scale = float(data.get("scale", 1.5))
        shift = data.get("shift_mm", [0, 0, 0])
        shift = tuple(float(v) for v in shift)
        return generate_scenario_tumor(case_id, scale_factor=scale, shift_mm=shift)
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc))
    except Exception as exc:
        logger.exception("VTK scenario tumor failed")
        raise HTTPException(500, f"VTK scenario tumor generation failed: {exc}")


@router.post("/upload-nifti")
async def upload_nifti_pair(
    ct_file: UploadFile = File(...),
    label_file: UploadFile = File(...),
    _user: User | None = Depends(get_optional_user),
):
    """Upload a matching CT .nii.gz volume and Label .nii.gz mask pair.

    The backend reads both files via Python VTK, strictly validates matching
    dimensions, spacing, origin, and coordinate matrices, and extracts 3D
    surfaces for Pancreas (label=1) and Cancer/Tumor (label=2) in physical
    coordinates.

    Returns the JSON 3D mesh payload consumed by the React/R3F TwinMeshViewer.
    """
    _ALLOWED = {".nii", ".gz"}
    ct_ext = Path(ct_file.filename or "").suffix.lower()
    lbl_ext = Path(label_file.filename or "").suffix.lower()
    if ct_ext not in _ALLOWED or lbl_ext not in _ALLOWED:
        raise HTTPException(415, "Only NIfTI files (.nii or .nii.gz) are supported for 3D extraction")

    tmp_dir = Path(tempfile.mkdtemp(prefix="vtk_nifti_"))
    try:
        ct_path = tmp_dir / _safe_name(ct_file.filename or "ct.nii.gz")
        lbl_path = tmp_dir / _safe_name(label_file.filename or "label.nii.gz")

        with ct_path.open("wb") as out:
            while chunk := await ct_file.read(1024 * 1024):
                out.write(chunk)

        with lbl_path.open("wb") as out:
            while chunk := await label_file.read(1024 * 1024):
                out.write(chunk)

        from digital_twin.vtk_engine import process_uploaded_nifti_pair
        return process_uploaded_nifti_pair(str(ct_path), str(lbl_path))

    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("Failed to process uploaded NIfTI pair")
        raise HTTPException(500, f"NIfTI pair processing failed: {exc}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post("/upload-folder")
async def upload_patient_folder(
    files: list[UploadFile] = File(...),
    _user: User | None = Depends(get_optional_user),
):
    """Upload a single patient case folder containing imagesTr, labelsTr, and/or meshes.npz.

    The endpoint reconstructs the folder layout in a temporary workspace, scans for:
      - imagesTr/*.nii.gz (CT volume)
      - labelsTr/*.nii.gz (Segmentation label)
      - meshes.npz (Pre-extracted mesh arrays)

    Performs Python VTK surface extraction for Pancreas (label=1) and Cancer/Tumor (label=2),
    loads NPZ mesh data if present, and returns a single combined 3D patient payload.
    """
    if not files:
        raise HTTPException(400, "No files uploaded in the patient folder.")

    logger.info("=== API /upload-folder: Received %d files ===", len(files))
    tmp_dir = Path(tempfile.mkdtemp(prefix="patient_folder_"))
    try:
        for file_item in files:
            fname = file_item.filename or "file.dat"
            clean_rel = fname.replace("\\", "/").lstrip("/")
            parts = [p for p in clean_rel.split("/") if p and p != ".."]
            dest_path = tmp_dir.joinpath(*parts)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            with dest_path.open("wb") as out:
                while chunk := await file_item.read(1024 * 1024):
                    out.write(chunk)

        from digital_twin.vtk_engine import process_uploaded_patient_folder
        return process_uploaded_patient_folder(str(tmp_dir))

    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        logger.exception("Failed to process uploaded patient case folder")
        raise HTTPException(500, f"Patient folder processing failed: {exc}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


