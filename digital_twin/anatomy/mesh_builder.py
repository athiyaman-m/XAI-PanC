"""Surface extraction from segmentation masks -> patient-specific 3D mesh.

The mesh is extracted *directly from the mask in the canonical voxel grid* and
then mapped to world (RAS, mm) via the PatientVolume affine, so the resulting
surface already carries correct physical dimensions and shares the same
coordinate system as every other structure (tumour, vessels, organs).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    from skimage.measure import marching_cubes
    _HAS_MC = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_MC = False

from .coordinate import voxel_to_world
from .models import PatientVolume
from .mesh_utils import decimate_mesh, laplacian_smooth, keep_largest_component


def mesh_from_mask(mask: np.ndarray, pv: PatientVolume,
                   decimate_to: Optional[int] = None,
                   smooth_iterations: int = 3) -> dict:
    """Marching-Cubes mesh of a binary mask, placed in world (RAS, mm) coords.

    Returns {"vertices": (N,3) world mm, "faces": (M,3) int32,
             "bounding_box": (min,max), "center": (3,), "dimensions_mm": (3,),
             "transform": 4x4 (identity in world frame),
             "voxel_to_world": pv voxel_to_world,
             "vertex_count": N, "face_count": M}
    """
    if not _HAS_MC:
        raise ImportError("scikit-image marching_cubes is required for mesh extraction")
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        raise ValueError("empty mask: cannot extract surface")
    # marching_cubes returns vertices in array-axis order (axis0, axis1, axis2)
    # == (slice/z, row/y, col/x) in VOXEL (index) units. The patient voxel->world
    # affine is defined over (col, row, slice) = (x, y, z), so we reorder to
    # (x, y, z) and apply the affine ONCE. Extracting in index units and letting
    # the affine do the scaling avoids any double-application of voxel spacing,
    # which was the root cause of the old mm/cm/scale mismatch -> floating dot.
    verts_vx, faces, _, _ = marching_cubes(mask.astype(np.float64), level=0.5,
                                           spacing=(1.0, 1.0, 1.0),
                                           gradient_direction='ascent')
    verts_vxyz = verts_vx[:, [2, 1, 0]]  # (z,y,x) -> (x,y,z) voxel indices
    verts_world = voxel_to_world(verts_vxyz, pv.voxel_to_world)
    verts, faces = keep_largest_component(verts_world, faces)
    if decimate_to and len(verts) > decimate_to:
        verts, faces = decimate_mesh(verts, faces, target_verts=decimate_to)
    if smooth_iterations:
        verts = laplacian_smooth(verts, faces, iterations=smooth_iterations, factor=0.4)
    info = _mesh_info(verts, faces)
    info["voxel_to_world"] = pv.voxel_to_world
    info["source"] = "marching_cubes"
    return info


def mesh_from_labelmap(mask: np.ndarray, pv: PatientVolume, label: int = 1,
                       **kwargs) -> dict:
    binary = mask == label
    return mesh_from_mask(binary, pv, **kwargs)


def mesh_info(verts: np.ndarray, faces: np.ndarray) -> dict:
    if len(verts) == 0:
        return {"bounding_box": (np.zeros(3), np.zeros(3)), "center": np.zeros(3),
                "dimensions_mm": (0.0, 0.0, 0.0), "vertex_count": 0, "face_count": 0}
    return _mesh_info(verts, faces)


def _mesh_info(verts: np.ndarray, faces: np.ndarray) -> dict:
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    mi = verts.min(axis=0)
    ma = verts.max(axis=0)
    dims = (float(ma[0] - mi[0]), float(ma[1] - mi[1]), float(ma[2] - mi[2]))
    center = (mi + ma) / 2.0
    # surface area
    try:
        a = verts[faces[:, 0]]; b = verts[faces[:, 1]]; c = verts[faces[:, 2]]
        area = float(np.sum(np.linalg.norm(np.cross(b - a, c - a), axis=1)) / 2.0)
    except Exception:
        area = 0.0
    # volume (signed tetrahedra)
    try:
        v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        vol = float(np.abs(np.sum(np.einsum('ij,ij->i', np.cross(v0, v1), v2))) / 6.0)
    except Exception:
        vol = 0.0
    return {
        "vertices": verts,
        "faces": faces,
        "bounding_box": (mi, ma),
        "center": center,
        "dimensions_mm": dims,
        "transform": np.eye(4),
        "volume_mm3": vol,
        "surface_area_mm2": area,
        "vertex_count": int(len(verts)),
        "face_count": int(len(faces)),
    }
