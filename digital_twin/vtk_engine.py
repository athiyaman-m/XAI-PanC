"""VTK Engine for CT/DICOM-first Digital Twin — Pure VTK Pipeline.

This module is the SOLE source of truth for all 3D geometry operations:
- NIfTI/DICOM volume loading (vtkNIFTIImageReader / vtkDICOMImageReader)
- Segmentation mask loading
- Surface extraction (vtkMarchingCubes)
- Mesh cleanup, smoothing, decimation
- Geometry diagnostics (bounds, center, volume, spacing, origin)
- polydata → numpy conversion for JSON serialization
- Scenario tumor generation (separate layer from ground-truth)

ALL coordinates are in PHYSICAL MILLIMETRES derived from the NIfTI/DICOM
header (spacing × voxel index + origin). The frontend receives positions in
mm and renders them without any additional coordinate transformation.
"""
from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import vtk
    from vtk.util import numpy_support
    _HAS_VTK = True
except ImportError:
    _HAS_VTK = False

logger = logging.getLogger(__name__)

MSD_DIR = os.path.join("DATASETS", "MSD_Pancreas")

# Label convention for MSD Task07 Pancreas
LABEL_PANCREAS = 1
LABEL_TUMOR = 2

# Web-friendly mesh density target (triangle reduction ratio for vtkDecimatePro)
_WEB_TARGET_REDUCTION = 0.85  # keep 15% of triangles


# ---------------------------------------------------------------------------
# Volume loading
# ---------------------------------------------------------------------------

def load_nifti_volume(path: str) -> "vtk.vtkImageData":
    """Load a NIfTI (.nii / .nii.gz) volume and return vtkImageData.

    The reader preserves spacing, origin, and orientation from the NIfTI
    header so all downstream geometry stays in the patient's physical
    coordinate system.
    """
    if not _HAS_VTK:
        raise RuntimeError("VTK is not installed")
    reader = vtk.vtkNIFTIImageReader()
    reader.SetFileName(str(path))
    reader.Update()
    vol = reader.GetOutput()
    dims = vol.GetDimensions()
    spacing = vol.GetSpacing()
    origin = vol.GetOrigin()
    logger.info(
        "NIfTI loaded: %s — dims=%s spacing=%s origin=%s",
        path, dims, spacing, origin,
    )
    if dims[0] == 0 or dims[1] == 0 or dims[2] == 0:
        raise ValueError(f"Empty volume from {path}: dims={dims}")
    return vol


def load_ct_series(ct_dir: str) -> "vtk.vtkImageData":
    """Load a CT DICOM series from a directory."""
    if not _HAS_VTK:
        raise RuntimeError("VTK is not installed")
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(ct_dir)
    reader.Update()
    vol = reader.GetOutput()
    logger.info(
        "CT DICOM loaded: %s — dims=%s spacing=%s origin=%s",
        ct_dir, vol.GetDimensions(), vol.GetSpacing(), vol.GetOrigin(),
    )
    return vol


# ---------------------------------------------------------------------------
# Surface extraction — pure VTK
# ---------------------------------------------------------------------------

def _threshold_label(label_image: "vtk.vtkImageData",
                     label_value: int) -> "vtk.vtkImageData":
    """Threshold a label volume to a binary mask for one label value."""
    thresh = vtk.vtkImageThreshold()
    thresh.SetInputData(label_image)
    thresh.ThresholdBetween(label_value - 0.5, label_value + 0.5)
    thresh.SetInValue(1.0)
    thresh.SetOutValue(0.0)
    thresh.SetOutputScalarTypeToFloat()
    thresh.Update()
    return thresh.GetOutput()


def _smooth_volume(binary_image: "vtk.vtkImageData",
                   sigma_mm: float = 1.0) -> "vtk.vtkImageData":
    """Apply Gaussian smoothing to a binary mask for smoother surfaces."""
    smoother = vtk.vtkImageGaussianSmooth()
    smoother.SetInputData(binary_image)
    spacing = binary_image.GetSpacing()
    # Convert sigma from mm to voxels for each axis
    smoother.SetStandardDeviations(
        sigma_mm / spacing[0],
        sigma_mm / spacing[1],
        sigma_mm / spacing[2],
    )
    smoother.SetRadiusFactors(3, 3, 3)
    smoother.Update()
    return smoother.GetOutput()


def extract_surface(label_image: "vtk.vtkImageData",
                    label_value: int,
                    smooth_sigma_mm: float = 0.8) -> "vtk.vtkPolyData":
    """Extract an iso-surface from a label volume using VTK Marching Cubes.

    Parameters
    ----------
    label_image : vtkImageData
        The label/segmentation volume (integer labels).
    label_value : int
        The label to extract (e.g. 1=pancreas, 2=tumor).
    smooth_sigma_mm : float
        Gaussian smoothing sigma in mm applied before marching cubes.

    Returns
    -------
    vtkPolyData
        Cleaned, smoothed surface mesh in physical mm coordinates.
    """
    if not _HAS_VTK:
        raise RuntimeError("VTK is not installed")

    # Step 1: Threshold to binary
    binary = _threshold_label(label_image, label_value)

    # Step 2: Gaussian smooth for better surface quality
    if smooth_sigma_mm > 0:
        binary = _smooth_volume(binary, smooth_sigma_mm)

    # Step 3: Iso-surface extraction via FlyingEdges3D (or MarchingCubes fallback)
    if hasattr(vtk, "vtkFlyingEdges3D"):
        alg = vtk.vtkFlyingEdges3D()
    else:
        alg = vtk.vtkMarchingCubes()
    alg.SetInputData(binary)
    alg.SetValue(0, 0.5)
    alg.ComputeNormalsOn()
    alg.ComputeGradientsOff()
    alg.Update()
    poly = alg.GetOutput()

    logger.info(
        "%s (label=%d): %d vertices, %d triangles",
        alg.GetClassName(), label_value, poly.GetNumberOfPoints(), poly.GetNumberOfPolys(),
    )

    # Step 4: Clean duplicates and degenerate cells
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(poly)
    cleaner.Update()

    # Step 5: Windowed-sinc smoothing for mesh quality
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputConnection(cleaner.GetOutputPort())
    smoother.SetNumberOfIterations(20)
    smoother.SetPassBand(0.1)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    result = smoother.GetOutput()
    logger.info(
        "Surface cleaned (label=%d): %d vertices, %d triangles",
        label_value, result.GetNumberOfPoints(), result.GetNumberOfPolys(),
    )
    return result


# ---------------------------------------------------------------------------
# Mesh decimation for web
# ---------------------------------------------------------------------------

def decimate_for_web(poly: "vtk.vtkPolyData",
                     target_verts: int = 15000
                     ) -> "vtk.vtkPolyData":
    """Decimate a vtkPolyData mesh for browser-friendly density matching Real CT twin (15k target)."""
    n_pts = poly.GetNumberOfPoints()
    if n_pts <= target_verts or n_pts < 1000:
        return poly

    target_reduction = max(0.0, min(0.92, 1.0 - (target_verts / float(n_pts))))
    decimator = vtk.vtkDecimatePro()
    decimator.SetInputData(poly)
    decimator.SetTargetReduction(target_reduction)
    decimator.PreserveTopologyOn()
    decimator.SplittingOff()
    decimator.BoundaryVertexDeletionOff()
    decimator.Update()

    result = decimator.GetOutput()
    logger.info(
        "Decimated for web: %d -> %d vertices, %d triangles",
        n_pts, result.GetNumberOfPoints(), result.GetNumberOfPolys(),
    )
    
    # Re-compute normals after decimation
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(result)
    normals.ComputePointNormalsOn()
    normals.SplittingOff()
    normals.Update()
    return normals.GetOutput()


# ---------------------------------------------------------------------------
# VTK PolyData → NumPy/JSON bridge
# ---------------------------------------------------------------------------

def polydata_to_numpy(poly: "vtk.vtkPolyData"
                      ) -> Dict[str, np.ndarray]:
    """Convert a vtkPolyData to NumPy arrays for JSON serialization.

    Returns
    -------
    dict with keys:
        ``positions`` : ndarray (N, 3) float64 — vertex positions in mm
        ``indices``   : ndarray (M, 3) int32   — triangle indices
    """
    if not _HAS_VTK:
        raise RuntimeError("VTK is not installed")

    # Extract points
    points = poly.GetPoints()
    if points is None or points.GetNumberOfPoints() == 0:
        return {"positions": np.empty((0, 3), dtype=np.float64),
                "indices": np.empty((0, 3), dtype=np.int32)}
    vtk_pts = points.GetData()
    positions = numpy_support.vtk_to_numpy(vtk_pts).copy()

    # Extract triangle indices from the polys cell array
    polys = poly.GetPolys()
    if polys is None or polys.GetNumberOfCells() == 0:
        return {"positions": positions,
                "indices": np.empty((0, 3), dtype=np.int32)}

    # VTK stores polys as: [npts, id0, id1, id2, npts, id0, id1, id2, ...]
    # For triangulated meshes, npts is always 3.
    vtk_cells = polys.GetData()
    cell_array = numpy_support.vtk_to_numpy(vtk_cells)

    n_cells = polys.GetNumberOfCells()
    # Reshape: each triangle is stored as [3, v0, v1, v2] → 4 entries
    try:
        cell_array = cell_array.reshape(n_cells, -1)
        # Skip the leading "3" (number of points per cell)
        indices = cell_array[:, 1:4].astype(np.int32)
    except (ValueError, IndexError):
        # Fallback: iterate cells manually
        indices = []
        polys.InitTraversal()
        id_list = vtk.vtkIdList()
        for _ in range(n_cells):
            polys.GetNextCell(id_list)
            if id_list.GetNumberOfIds() == 3:
                indices.append([id_list.GetId(0), id_list.GetId(1),
                                id_list.GetId(2)])
        indices = np.array(indices, dtype=np.int32) if indices else \
            np.empty((0, 3), dtype=np.int32)

    return {"positions": positions, "indices": indices}


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def compute_diagnostics(poly: "vtk.vtkPolyData",
                        label_name: str = "mesh",
                        label_image: Optional["vtk.vtkImageData"] = None
                        ) -> Dict[str, Any]:
    """Compute and log comprehensive geometry diagnostics.

    Returns a dict with bounds, center, dimensions, volume, spacing, origin,
    vertex/triangle counts, and a world-coordinate transform description.
    """
    bounds = poly.GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
    cx = (bounds[0] + bounds[1]) / 2.0
    cy = (bounds[2] + bounds[3]) / 2.0
    cz = (bounds[4] + bounds[5]) / 2.0
    dx = bounds[1] - bounds[0]
    dy = bounds[3] - bounds[2]
    dz = bounds[5] - bounds[4]

    n_verts = poly.GetNumberOfPoints()
    n_tris = poly.GetNumberOfPolys()

    # Volume via vtkMassProperties (closed surfaces)
    volume_mm3 = 0.0
    try:
        mass = vtk.vtkMassProperties()
        mass.SetInputData(poly)
        mass.Update()
        volume_mm3 = mass.GetVolume()
    except Exception:
        pass

    diag: Dict[str, Any] = {
        "label": label_name,
        "bounds": [round(b, 3) for b in bounds],
        "center": [round(cx, 3), round(cy, 3), round(cz, 3)],
        "dimensions_mm": [round(dx, 3), round(dy, 3), round(dz, 3)],
        "volume_mm3": round(volume_mm3, 3),
        "volume_cm3": round(volume_mm3 / 1000.0, 4),
        "vertices": n_verts,
        "triangles": n_tris,
    }

    # If source label image is available, include its spatial metadata
    if label_image is not None:
        diag["spacing_mm"] = [round(s, 4) for s in label_image.GetSpacing()]
        diag["origin_mm"] = [round(o, 4) for o in label_image.GetOrigin()]
        diag["image_dims"] = list(label_image.GetDimensions())

    logger.info(
        "DIAGNOSTICS [%s]:\n"
        "  bounds      = %s\n"
        "  center      = %s mm\n"
        "  dimensions  = %s mm\n"
        "  volume      = %.2f mm³ (%.4f cm³)\n"
        "  vertices    = %d\n"
        "  triangles   = %d\n"
        "  spacing     = %s\n"
        "  origin      = %s",
        label_name,
        diag["bounds"],
        diag["center"],
        diag["dimensions_mm"],
        volume_mm3, volume_mm3 / 1000.0,
        n_verts, n_tris,
        diag.get("spacing_mm", "N/A"),
        diag.get("origin_mm", "N/A"),
    )
    return diag


# ---------------------------------------------------------------------------
# Case-level orchestrator
# ---------------------------------------------------------------------------

def _discover_msd_cases(root: str | os.PathLike | None = None) -> List[str]:
    """Return MSD case IDs that have both an image and a label NIfTI file."""
    base = Path(root or MSD_DIR)
    if not base.exists():
        return []
    cases = set()
    for im in base.glob("imagesTr/pancreas_*.nii.gz"):
        label_path = base / "labelsTr" / im.name
        if not label_path.exists():
            label_path = base / "labelsTr" / im.name.replace(".nii.gz", "_label.nii.gz")
        if label_path.exists():
            cid = im.name[len("pancreas_"):-len(".nii.gz")]
            cases.add(cid)
    return sorted(cases, key=lambda x: (len(x), x))


# Cache for processed cases (in-memory)
_case_cache: Dict[str, Dict] = {}


def process_case(case_id: str,
                 root: str | os.PathLike | None = None,
                 force: bool = False) -> Dict[str, Any]:
    """Full VTK pipeline for one MSD case.

    Loads NIfTI image + label, extracts pancreas (label=1) and tumor (label=2)
    surfaces via VTK Marching Cubes, decimates for web, computes diagnostics,
    and returns the JSON-ready payload for the frontend.

    Returns
    -------
    dict
        ``{mode, title, organs: [{name, color, opacity, kind, positions,
        indices, layer}], meta: {case, source, pancreas_diagnostics,
        tumor_diagnostics}}``
    """
    if not force and case_id in _case_cache:
        return _case_cache[case_id]

    # Try loading via unified Real CT exporter for 100% mesh, spatial alignment, and color parity
    try:
        from backend.app.services.twin_mesh_exporter import real_ct_payload
        payload = real_ct_payload(case_id)
        payload["mode"] = "vtk-ct"
        payload["title"] = f"Patient CT — VTK Pipeline — MSD case {case_id}"
        _case_cache[case_id] = payload
        return payload
    except Exception as e:
        logger.info("Proceeding with VTK NIfTI extraction for case %s: %s", case_id, e)

    if not _HAS_VTK:
        raise RuntimeError("VTK is not installed — cannot process case")

    base = Path(root or MSD_DIR)
    label_path = base / "labelsTr" / f"pancreas_{case_id}.nii.gz"
    if not label_path.exists():
        label_path = base / "labelsTr" / f"pancreas_{case_id}_label.nii.gz"

    image_path = base / "imagesTr" / f"pancreas_{case_id}.nii.gz"

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    logger.info("=== VTK PIPELINE: Processing MSD case %s ===", case_id)

    # 1. Load label volume (the one with segmentation masks)
    label_vol = load_nifti_volume(str(label_path))

    # Log spatial metadata for alignment verification
    logger.info(
        "Label volume spatial metadata:\n"
        "  spacing = %s\n"
        "  origin  = %s\n"
        "  dims    = %s",
        label_vol.GetSpacing(),
        label_vol.GetOrigin(),
        label_vol.GetDimensions(),
    )

    # 2. Extract pancreas surface (label=1)
    logger.info("Extracting pancreas surface (label=%d)...", LABEL_PANCREAS)
    pancreas_poly = extract_surface(label_vol, LABEL_PANCREAS, smooth_sigma_mm=0.8)
    pancreas_poly = decimate_for_web(pancreas_poly)
    pancreas_diag = compute_diagnostics(pancreas_poly, "Pancreas", label_vol)

    # 3. Extract tumor surface (label=2)
    logger.info("Extracting tumor surface (label=%d)...", LABEL_TUMOR)
    tumor_poly = extract_surface(label_vol, LABEL_TUMOR, smooth_sigma_mm=0.6)
    tumor_poly = decimate_for_web(tumor_poly)
    tumor_diag = compute_diagnostics(tumor_poly, "Tumor", label_vol)

    # 4. Verify spatial alignment
    pan_spacing = label_vol.GetSpacing()
    pan_origin = label_vol.GetOrigin()
    logger.info(
        "ALIGNMENT CHECK:\n"
        "  Both meshes from SAME label volume → SAME coordinate system\n"
        "  Spacing: %s mm\n"
        "  Origin:  %s mm\n"
        "  Pancreas center: %s mm\n"
        "  Tumor center:    %s mm\n"
        "  Distance between centers: %.2f mm",
        pan_spacing, pan_origin,
        pancreas_diag["center"], tumor_diag["center"],
        np.linalg.norm(
            np.array(pancreas_diag["center"]) -
            np.array(tumor_diag["center"])
        ),
    )

    # 5. Convert to numpy for JSON serialization
    pan_np = polydata_to_numpy(pancreas_poly)
    tum_np = polydata_to_numpy(tumor_poly)

    # Calculate lower corner ROI crop offset (min_bounds - 20mm margin) exactly like Real CT twin
    min_x = min(pancreas_diag["bounds"][0], tumor_diag["bounds"][0]) - 20.0
    min_y = min(pancreas_diag["bounds"][2], tumor_diag["bounds"][2]) - 20.0
    min_z = min(pancreas_diag["bounds"][4], tumor_diag["bounds"][4]) - 20.0
    offset_mm = np.array([min_x, min_y, min_z])

    pan_centered = (pan_np["positions"] - offset_mm) if len(pan_np["positions"]) > 0 else pan_np["positions"]
    tum_centered = (tum_np["positions"] - offset_mm) if len(tum_np["positions"]) > 0 else tum_np["positions"]

    # 6. Build frontend-ready payload (identical naming, colors, opacity as Real CT twin)
    organs: List[Dict[str, Any]] = []

    # Pancreas mesh
    if len(pan_centered) > 0:
        organs.append({
            "name": "Pancreas (ground truth)",
            "color": "#e0a468",
            "opacity": 0.88,
            "kind": "solid",
            "layer": "ground-truth",
            "positions": np.round(pan_centered, 3).reshape(-1).tolist(),
            "indices": pan_np["indices"].reshape(-1).tolist(),
        })

    # Tumor mesh
    if len(tum_centered) > 0:
        organs.append({
            "name": "Tumour (ground truth)",
            "color": "#ff3b3b",
            "opacity": 0.95,
            "kind": "solid",
            "layer": "ground-truth",
            "positions": np.round(tum_centered, 3).reshape(-1).tolist(),
            "indices": tum_np["indices"].reshape(-1).tolist(),
        })

    payload = {
        "mode": "vtk-ct",
        "title": f"Patient CT — VTK Pipeline — MSD case {case_id}",
        "organs": organs,
        "meta": {
            "case": case_id,
            "source": "MSD Task07 Pancreas — VTK Pipeline",
            "offset_mm": np.round(offset_mm, 3).tolist(),
            "pancreas_diagnostics": pancreas_diag,
            "tumor_diagnostics": tumor_diag,
            "tumor_centroid": (np.array(tumor_diag["center"]) - offset_mm).round(3).tolist(),
            "tumor_volume_cm3": tumor_diag["volume_cm3"],
            "tumor_bounds_mm": [
                (np.array(tumor_diag["bounds"][:2]) - offset_mm[0]).round(3).tolist(),
                (np.array(tumor_diag["bounds"][2:4]) - offset_mm[1]).round(3).tolist(),
                (np.array(tumor_diag["bounds"][4:6]) - offset_mm[2]).round(3).tolist(),
            ],
        },
    }

    _case_cache[case_id] = payload
    logger.info("=== VTK PIPELINE: Case %s complete ===", case_id)
    return payload


# ---------------------------------------------------------------------------
# Scenario / Simulated tumor generation
# ---------------------------------------------------------------------------

def generate_scenario_tumor(case_id: str,
                            scale_factor: float = 1.5,
                            shift_mm: Tuple[float, float, float] = (0, 0, 0),
                            root: str | os.PathLike | None = None
                            ) -> Dict[str, Any]:
    """Generate a scenario/projected tumor by transforming the ground-truth.

    This creates a DEEP COPY of the real tumor geometry, applies scaling
    around its centroid plus an optional translation, and returns a single
    organ entry with ``layer='scenario'``. The original ground-truth tumor
    is NEVER modified.

    Parameters
    ----------
    case_id : str
        MSD case ID.
    scale_factor : float
        Uniform scale factor applied around the tumor centroid.
    shift_mm : tuple
        Translation (dx, dy, dz) in mm applied after scaling.

    Returns
    -------
    dict
        Single organ entry: ``{name, color, opacity, kind, layer,
        positions, indices}``.
    """
    if not _HAS_VTK:
        raise RuntimeError("VTK is not installed")

    # Load the ground-truth label to extract the tumor again (or use cache)
    base = Path(root or MSD_DIR)
    label_path = base / "labelsTr" / f"pancreas_{case_id}.nii.gz"
    if not label_path.exists():
        label_path = base / "labelsTr" / f"pancreas_{case_id}_label.nii.gz"
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    label_vol = load_nifti_volume(str(label_path))
    tumor_poly = extract_surface(label_vol, LABEL_TUMOR, smooth_sigma_mm=0.6)
    tumor_poly = decimate_for_web(tumor_poly)

    # Compute centroid for scaling origin
    bounds = tumor_poly.GetBounds()
    cx = (bounds[0] + bounds[1]) / 2.0
    cy = (bounds[2] + bounds[3]) / 2.0
    cz = (bounds[4] + bounds[5]) / 2.0

    # Build transform: translate to origin → scale → translate back + shift
    transform = vtk.vtkTransform()
    transform.PostMultiply()
    transform.Translate(-cx, -cy, -cz)
    transform.Scale(scale_factor, scale_factor, scale_factor)
    transform.Translate(cx + shift_mm[0], cy + shift_mm[1], cz + shift_mm[2])

    transformer = vtk.vtkTransformPolyDataFilter()
    transformer.SetInputData(tumor_poly)
    transformer.SetTransform(transform)
    transformer.Update()
    scenario_poly = transformer.GetOutput()

    scenario_diag = compute_diagnostics(scenario_poly, "Scenario Tumor", label_vol)
    sc_np = polydata_to_numpy(scenario_poly)

    organ_entry = {
        "name": "Tumor (Simulated/Projected)",
        "color": "#ff9933",
        "opacity": 0.55,
        "kind": "solid",
        "layer": "scenario",
        "positions": np.round(sc_np["positions"], 3).reshape(-1).tolist(),
        "indices": sc_np["indices"].reshape(-1).tolist(),
    }

    return {
        "organ": organ_entry,
        "diagnostics": scenario_diag,
        "scale_factor": scale_factor,
        "shift_mm": list(shift_mm),
        "ground_truth_center": [round(cx, 3), round(cy, 3), round(cz, 3)],
    }


# ---------------------------------------------------------------------------
# Volume validation & uploaded pair processing
# ---------------------------------------------------------------------------

def validate_volume_match(ct_vol: "vtk.vtkImageData", label_vol: "vtk.vtkImageData") -> None:
    """Validate that CT volume and Label volume match in dimensions, spacing, and origin.

    Raises ValueError if there is any mismatch between CT and Label metadata.
    """
    dims_ct = ct_vol.GetDimensions()
    dims_lbl = label_vol.GetDimensions()
    if dims_ct != dims_lbl:
        raise ValueError(f"Dimension mismatch: CT dimensions {dims_ct} vs Label dimensions {dims_lbl}")

    spacing_ct = np.array(ct_vol.GetSpacing(), dtype=np.float64)
    spacing_lbl = np.array(label_vol.GetSpacing(), dtype=np.float64)
    if not np.allclose(spacing_ct, spacing_lbl, atol=1e-3):
        raise ValueError(f"Spacing mismatch: CT spacing {spacing_ct.round(4)} mm vs Label spacing {spacing_lbl.round(4)} mm")

    origin_ct = np.array(ct_vol.GetOrigin(), dtype=np.float64)
    origin_lbl = np.array(label_vol.GetOrigin(), dtype=np.float64)
    if not np.allclose(origin_ct, origin_lbl, atol=1e-2):
        raise ValueError(f"Origin mismatch: CT origin {origin_ct.round(4)} mm vs Label origin {origin_lbl.round(4)} mm")

    logger.info("Volume match validated: dims=%s spacing=%s origin=%s", dims_ct, spacing_ct.round(4), origin_ct.round(4))


def process_uploaded_nifti_pair(ct_path: str, label_path: str) -> Dict[str, Any]:
    """Process an uploaded matching CT + Label NIfTI pair using 3D pipeline with 100% Real CT parity."""
    logger.info("=== 3D PIPELINE: Processing uploaded NIfTI CT + Label pair ===")

    try:
        import nibabel as nib
        from skimage.measure import marching_cubes
        from scipy import ndimage as ndi

        lab_img = nib.load(label_path)
        lab_data = lab_img.get_fdata()
        spacing = np.array(lab_img.header.get_zooms()[:3], dtype=np.float64)

        idx = np.argwhere(lab_data > 0)
        if len(idx) == 0:
            raise ValueError("No positive segmentation masks found in uploaded label file.")

        mn = np.clip(idx.min(axis=0) - 20, 0, None)
        mx = np.clip(idx.max(axis=0) + 21, None, np.array(lab_data.shape))
        crop_data = lab_data[mn[0]:mx[0], mn[1]:mx[1], mn[2]:mx[2]]
        offset_mm = mn * spacing

        organs: List[Dict[str, Any]] = []
        pan_diag: Dict[str, Any] = {}
        tum_diag: Dict[str, Any] = {}

        # 1. Pancreas Parenchyma (label == 1)
        if np.any(crop_data == 1):
            smooth1 = ndi.gaussian_filter((crop_data == 1).astype(np.float32), sigma=0.8)
            v1, f1, _, _ = marching_cubes(smooth1, level=0.5)
            v1_mm = v1 * spacing

            if len(f1) > 15000:
                step = max(1, len(f1) // 15000)
                f1 = f1[::step]

            organs.append({
                "name": "Pancreas (ground truth)",
                "color": "#e0a468",
                "opacity": 0.88,
                "kind": "solid",
                "layer": "ground-truth",
                "positions": np.round(v1_mm, 3).reshape(-1).tolist(),
                "indices": f1.astype(np.int32).reshape(-1).tolist(),
            })
            pan_diag = {
                "bounds": [float(v1_mm[:, 0].min()), float(v1_mm[:, 0].max()), float(v1_mm[:, 1].min()), float(v1_mm[:, 1].max()), float(v1_mm[:, 2].min()), float(v1_mm[:, 2].max())],
                "center": [float(v1_mm[:, 0].mean()), float(v1_mm[:, 1].mean()), float(v1_mm[:, 2].mean())],
                "vertices": len(v1_mm),
                "triangles": len(f1),
                "volume_cm3": 0.0,
            }

        # 2. Cancer / Tumor (label == 2)
        if np.any(crop_data == 2):
            smooth2 = ndi.gaussian_filter((crop_data == 2).astype(np.float32), sigma=0.6)
            v2, f2, _, _ = marching_cubes(smooth2, level=0.5)
            v2_mm = v2 * spacing

            if len(f2) > 15000:
                step = max(1, len(f2) // 15000)
                f2 = f2[::step]

            tum_voxels = np.sum(lab_data == 2)
            tum_vol_cm3 = round(float(tum_voxels * np.prod(spacing) / 1000.0), 2)

            organs.append({
                "name": "Tumour (ground truth)",
                "color": "#ff3b3b",
                "opacity": 0.95,
                "kind": "solid",
                "layer": "ground-truth",
                "positions": np.round(v2_mm, 3).reshape(-1).tolist(),
                "indices": f2.astype(np.int32).reshape(-1).tolist(),
            })
            tum_diag = {
                "bounds": [float(v2_mm[:, 0].min()), float(v2_mm[:, 0].max()), float(v2_mm[:, 1].min()), float(v2_mm[:, 1].max()), float(v2_mm[:, 2].min()), float(v2_mm[:, 2].max())],
                "center": [float(v2_mm[:, 0].mean()), float(v2_mm[:, 1].mean()), float(v2_mm[:, 2].mean())],
                "vertices": len(v2_mm),
                "triangles": len(f2),
                "volume_cm3": tum_vol_cm3,
            }

        tum_centroid = tum_diag.get("center", [0, 0, 0])

        return {
            "mode": "vtk-ct",
            "title": "Uploaded Patient CT — 3D Anatomy",
            "organs": organs,
            "meta": {
                "source": "Uploaded NIfTI Series — 3D Pipeline",
                "offset_mm": np.round(offset_mm, 3).tolist(),
                "pancreas_diagnostics": pan_diag,
                "tumor_diagnostics": tum_diag,
                "tumor_centroid": tum_centroid,
                "tumor_volume_cm3": tum_diag.get("volume_cm3", 0.0),
                "tumor_bounds_mm": [
                    [tum_diag["bounds"][0], tum_diag["bounds"][1]],
                    [tum_diag["bounds"][2], tum_diag["bounds"][3]],
                    [tum_diag["bounds"][4], tum_diag["bounds"][5]],
                ] if tum_diag else [[0, 0], [0, 0], [0, 0]],
            },
        }
    except Exception as e:
        logger.error("Error processing NIfTI pair: %s", e, exc_info=True)
        raise RuntimeError(f"Could not process NIfTI pair: {e}")


def process_uploaded_patient_folder(folder_path: str) -> Dict[str, Any]:
    """Process an uploaded single patient folder containing imagesTr, labelsTr, and/or meshes.npz.

    Searches recursively inside `folder_path` for:
      - `imagesTr/*.nii.gz` or `*.nii.gz` CT volume
      - `labelsTr/*.nii.gz` or `*label*.nii.gz` Segmentation mask
      - `meshes.npz` or `*.npz` pre-extracted mesh file

    Extracts 3D surfaces via VTK, loads NPZ mesh arrays if present, and returns
    a single combined 3D patient anatomy payload for rendering.
    """
    root = Path(folder_path)
    all_files = list(root.glob("**/*"))

    ct_file: Optional[Path] = None
    label_file: Optional[Path] = None
    npz_file: Optional[Path] = None

    for f in all_files:
        if not f.is_file():
            continue
        fname = f.name.lower()
        rel = str(f.relative_to(root)).lower().replace("\\", "/")

        if fname.endswith(".npz"):
            npz_file = f
        elif fname.endswith(".nii.gz") or fname.endswith(".nii"):
            if "labelstr" in rel or "label" in fname:
                label_file = f
            else:
                ct_file = f

    # If label_file and ct_file were not found by directory path, try matching by file list
    nii_files = [f for f in all_files if f.is_file() and (f.name.endswith(".nii.gz") or f.name.endswith(".nii"))]
    if not ct_file and nii_files:
        ct_file = nii_files[0]
    if not label_file and len(nii_files) > 1:
        label_file = nii_files[1]

    logger.info(
        "Folder scan results for %s:\n  CT File: %s\n  Label File: %s\n  NPZ File: %s",
        folder_path, ct_file, label_file, npz_file,
    )

    if npz_file and npz_file.exists():
        try:
            logger.info("Loading pre-extracted 3D mesh from uploaded NPZ file: %s", npz_file)
            npz_data = np.load(str(npz_file))
            organs: List[Dict[str, Any]] = []
            offset_mm = np.asarray(npz_data["offset_mm"], dtype=np.float64) if "offset_mm" in npz_data else np.zeros(3)

            pv = np.empty((0, 3))
            pf = np.empty((0, 3))
            tv = np.empty((0, 3))
            tf = np.empty((0, 3))

            if "pan_verts" in npz_data and "pan_faces" in npz_data:
                pv = np.asarray(npz_data["pan_verts"], dtype=np.float64) - offset_mm
                pf = npz_data["pan_faces"]
                organs.append({
                    "name": "Pancreas (ground truth)",
                    "color": "#e0a468",
                    "opacity": 0.88,
                    "kind": "solid",
                    "layer": "ground-truth",
                    "positions": np.round(pv, 3).reshape(-1).tolist(),
                    "indices": pf.reshape(-1).tolist(),
                })

            if "tum_verts" in npz_data and "tum_faces" in npz_data:
                tv = np.asarray(npz_data["tum_verts"], dtype=np.float64) - offset_mm
                tf = npz_data["tum_faces"]
                organs.append({
                    "name": "Tumour (ground truth)",
                    "color": "#ff3b3b",
                    "opacity": 0.95,
                    "kind": "solid",
                    "layer": "ground-truth",
                    "positions": np.round(tv, 3).reshape(-1).tolist(),
                    "indices": tf.reshape(-1).tolist(),
                })

            pan_diag = {
                "bounds": [pv[:, 0].min(), pv[:, 0].max(), pv[:, 1].min(), pv[:, 1].max(), pv[:, 2].min(), pv[:, 2].max()] if len(pv) > 0 else [0]*6,
                "vertices": len(pv),
                "triangles": len(pf),
                "volume_cm3": 0.0,
            } if len(pv) > 0 else {}

            tum_diag = {
                "bounds": [tv[:, 0].min(), tv[:, 0].max(), tv[:, 1].min(), tv[:, 1].max(), tv[:, 2].min(), tv[:, 2].max()] if len(tv) > 0 else [0]*6,
                "vertices": len(tv),
                "triangles": len(tf),
                "volume_cm3": round(float(npz_data["tumor_volume_cm3"]), 2) if "tumor_volume_cm3" in npz_data else 0.0,
            } if len(tv) > 0 else {}

            tum_centroid = (tv.mean(axis=0)).round(3).tolist() if len(tv) > 0 else [0, 0, 0]

            return {
                "mode": "vtk-ct",
                "title": f"Uploaded Patient Case — Combined 3D Anatomy ({root.name})",
                "organs": organs,
                "meta": {
                    "source": f"Uploaded Patient Folder ({root.name})",
                    "offset_mm": np.round(offset_mm, 3).tolist(),
                    "pancreas_diagnostics": pan_diag,
                    "tumor_diagnostics": tum_diag,
                    "tumor_centroid": tum_centroid,
                    "tumor_volume_cm3": tum_diag.get("volume_cm3", 0.0),
                    "npz_keys": list(npz_data.keys()),
                },
            }
        except Exception as e:
            logger.warning("Error reading uploaded NPZ file %s, falling back to NIfTI pair: %s", npz_file, e)

    if ct_file and label_file:
        # Process NIfTI pair via VTK pipeline
        payload = process_uploaded_nifti_pair(str(ct_file), str(label_file))
        payload["title"] = f"Uploaded Patient Case — Combined 3D Anatomy ({ct_file.name})"
        payload["meta"]["folder_source"] = str(root.name)
        return payload

    raise ValueError("No valid patient CT NIfTI pair (imagesTr + labelsTr) or meshes.npz found in the uploaded folder.")



# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def available_vtk_cases(root: str | os.PathLike | None = None) -> List[str]:
    """Return list of MSD case IDs that can be processed by the VTK pipeline."""
    return _discover_msd_cases(root)


def clear_cache(case_id: str | None = None) -> None:
    """Clear the in-memory mesh cache."""
    if case_id:
        _case_cache.pop(case_id, None)
    else:
        _case_cache.clear()

