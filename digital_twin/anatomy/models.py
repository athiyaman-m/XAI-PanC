"""Data model for the CT-based patient-specific digital twin.

These structures keep the data / model-generation layer strictly separate from
rendering (rendering lives in organ_renderer.py / the new CT twin renderer).

PatientCT -> PancreasModel -> (shared world frame) -> TumorModel
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .coordinate import MM, to_mm


@dataclass
class PatientVolume:
    """A CT/DICOM volume fully registered in its canonical physical (RAS, mm) frame.

    `image` is a (Z, Y, X) or (D, H, W) float32 array (post HU correction).
    `voxel_to_world` is the 4x4 affine mapping voxel indices -> world (RAS) mm.
    Everything else (origin, direction, spacing, dimensions) is derived from it
    but stored explicitly for diagnostics.
    """
    image: np.ndarray
    voxel_to_world: np.ndarray
    world_to_voxel: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float]
    direction: tuple[float, float, float, float, float, float, float, float, float]
    dimensions: tuple[int, int, int]  # voxels (D, H, W)
    physical_size_mm: tuple[float, float, float]
    modality: str = "CT"
    series_uid: str = ""
    source_path: str = ""

    @property
    def affine(self) -> np.ndarray:
        return self.voxel_to_world

    def world_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_corner, max_corner) of the volume in world mm."""
        d, h, w = self.dimensions
        corners = np.array([
            [0, 0, 0],
            [w, 0, 0],
            [0, h, 0],
            [w, h, 0],
            [0, 0, d],
            [w, 0, d],
            [0, h, d],
            [w, h, d],
        ], dtype=np.float64)
        from .coordinate import voxel_to_world
        world = voxel_to_world(corners, self.voxel_to_world)
        return world.min(axis=0), world.max(axis=0)

    def diagnose(self) -> dict:
        arr = np.asarray(self.image, dtype=np.float32)
        mn, mx = float(np.min(arr)), float(np.max(arr))
        lo, hi = float(np.percentile(arr, 1)), float(np.percentile(arr, 99))
        return {
            "modality": self.modality,
            "dimensions_voxels": list(self.dimensions),
            "spacing_mm": list(self.spacing),
            "origin_mm": list(self.origin),
            "direction": list(self.direction),
            "physical_size_mm": list(self.physical_size_mm),
            "hu_range": [mn, mx],
            "hu_window": [lo, hi],
            "series_uid": self.series_uid,
            "source_path": self.source_path,
        }


@dataclass
class PancreasModel:
    """Patient-specific pancreas mesh + mask, in the shared world frame."""
    mesh_vertices: np.ndarray        # (N, 3) world mm
    mesh_faces: np.ndarray           # (M, 3)
    mask: Optional[np.ndarray] = None       # voxel mask aligned to PatientVolume
    bounding_box: tuple[np.ndarray, np.ndarray] = (np.zeros(3), np.zeros(3))
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dimensions_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))  # identity in world frame
    principal_axis: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    regions: dict = field(default_factory=dict)      # {name: {"voxel_mask", "centre", "extent"}}
    source: str = "segmentation"  # or "atlas_fallback"
    voxel_to_world: np.ndarray = field(default_factory=lambda: np.eye(4))


@dataclass
class TumorState:
    """Single, central source of truth for tumor clinical parameters.

    All UI controls write here; the TumorEngine reads from here.
    """
    source: str = "SIMULATED"            # "SIMULATED" | "PATIENT_SEGMENTATION"
    location: str = "HEAD"               # HEAD | NECK | BODY | TAIL
    length_mm: float = 35.0
    width_mm: float = 28.0
    height_mm: float = 30.0
    shape: str = "irregular"             # spherical | ellipsoidal | irregular
    progression: float = 0.65            # 0..1 morphological complexity / growth
    invasion_level: float = 0.20         # 0..1 fraction protruding beyond pancreas
    vascular_involvement: bool = False
    # physical unit for everything is millimetres (no cm/m confusion)
    unit: str = MM

    def dimensions(self, unit: str = MM) -> tuple[float, float, float]:
        return (to_mm(self.length_mm, self.unit),
                to_mm(self.width_mm, self.unit),
                to_mm(self.height_mm, self.unit))

    def volume_mm3(self) -> float:
        # approximate as ellipsoid
        a, b, c = self.dimensions()
        return float((4.0 / 3.0) * np.pi * (a / 2.0) * (b / 2.0) * (c / 2.0))


@dataclass
class TumorModel:
    """Generated tumor geometry + measurements, in the same world frame as the pancreas."""
    mesh_vertices: np.ndarray
    mesh_faces: np.ndarray
    mask: Optional[np.ndarray] = None
    bounding_box: tuple[np.ndarray, np.ndarray] = (np.zeros(3), np.zeros(3))
    center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    dimensions_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    volume_mm3: float = 0.0
    max_diameter_mm: float = 0.0
    transform: np.ndarray = field(default_factory=lambda: np.eye(4))
    distances_mm: dict = field(default_factory=dict)   # {organ_name: min distance}
    source: str = "SIMULATED"
    label: str = "AI-Simulated Tumor"
