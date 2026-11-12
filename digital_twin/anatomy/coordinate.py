"""Canonical anatomical coordinate system + physical unit conversions.

The single source of truth for spatial mapping in the ExAI-PanC digital twin:

    voxel (col, row, slice)  <--voxel_to_world-->  world (RAS, mm)

**Conventions (shared by every module in this library):**

* The CT volume array is indexed `[k, j, i]` and stored in voxel-axis order
  ``(axis0, axis1, axis2) = (slice/z, row/y, col/x)``.
* ``PatientVolume.spacing`` is in the SAME voxel-axis order ``(sz, sy, sx)``,
  matching `nibabel.get_zooms()` and SimpleITK (so marching-cubes and
  scipy distance transforms consume it directly).
* ``voxel_to_world`` affine maps a voxel index ``(i=col, j=row, k=slice)`` to a
  world point in RAS (Right-Anterior-Superior, millimetres):
  ``world = affine @ [i, j, k, 1]``. Columns of the affine are the world x (col),
  y (row) and z (slice) axis directions scaled by ``(sx, sy, sz)``.

Everything downstream (pancreas mask, pancreas mesh, tumor mask/mesh, surrounding
anatomy, landmarks) lives in this same world/RAS-mm frame. No organ is ever
normalised to its own private [-1, 1] box without keeping the reversible
transform, which was the root cause of the old "floating dot" tumor.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------
MM = "mm"
CM = "cm"
METER = "m"

_UNIT_TO_MM = {
    MM: 1.0,
    CM: 10.0,
    METER: 1000.0,
}


def to_mm(value: float, unit: str = MM) -> float:
    """Convert a scalar physical value to millimetres."""
    factor = _UNIT_TO_MM.get(str(unit).lower())
    if factor is None:
        raise ValueError(f"unknown physical unit: {unit!r}")
    return float(value) * factor


def convert_units(value: float, from_unit: str, to_unit: str = MM) -> float:
    return to_mm(value, from_unit) / to_mm(1.0, to_unit)


# ---------------------------------------------------------------------------
# Voxel <-> world
# ---------------------------------------------------------------------------
def voxel_to_world(voxel_coords: np.ndarray, voxel_to_world: np.ndarray) -> np.ndarray:
    """Map voxel indices ``(col, row, slice)`` to world (RAS, mm) coordinates.

    ``voxel_coords`` is an (N, 3) array whose columns are (col=i, row=j,
    slice=k) in voxel-axis order matching ``PatientVolume.spacing``.
    """
    v = np.asarray(voxel_coords, dtype=np.float64)
    if v.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if v.ndim == 1:
        v = v.reshape(1, 3)
    homog = np.hstack([v, np.ones((len(v), 1), dtype=np.float64)])
    world = homog @ voxel_to_world.T
    return world[:, :3]


def world_to_voxel(world_coords: np.ndarray, world_to_voxel: np.ndarray) -> np.ndarray:
    w = np.asarray(world_coords, dtype=np.float64)
    if w.size == 0:
        return np.empty((0, 3), dtype=np.float64)
    if w.ndim == 1:
        w = w.reshape(1, 3)
    homog = np.hstack([w, np.ones((len(w), 1), dtype=np.float64)])
    return homog @ world_to_voxel.T


def voxel_bbox_mm(mask: np.ndarray, voxel_to_world: np.ndarray) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Axis-aligned bounding box (world, mm) of a binary voxel mask."""
    idx = np.argwhere(mask)
    if len(idx) == 0:
        return np.zeros((2, 3)), (0.0, 0.0, 0.0)
    corners = np.array([
        idx.min(axis=0),
        idx.max(axis=0) + 1,
    ], dtype=np.float64)
    world = voxel_to_world(corners, voxel_to_world)
    dims = (world[1] - world[0]).tolist()
    return world, (abs(dims[0]), abs(dims[1]), abs(dims[2]))


def affine_identity() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


def affine_from_spacing_origin_direction(spacing, origin, direction):
    """Build a 4x4 voxel->world (RAS, mm) affine from SimpleITK-style metadata.

    direction is a 9-tuple (row major) or 3x3 (world axes x voxel axes).
    """
    spacing = np.asarray(spacing, dtype=np.float64).reshape(3)
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    if direction is None:
        direction = np.eye(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3, 3)
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = d * spacing[:, None]
    affine[:3, 3] = origin
    return affine
