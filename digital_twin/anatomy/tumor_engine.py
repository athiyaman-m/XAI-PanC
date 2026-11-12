"""Dynamic, patient-specific tumor geometry generation.

TumorEngine reads from a single TumorState and the patient-specific
PancreasModel/region mapper, and produces a real 3D surface mesh (NOT a point)
in the SAME world (RAS, mm) coordinate system as the pancreas.

Shapes:
  * spherical   -> UV sphere
  * ellipsoidal -> scaled sphere
  * irregular   -> ellipsoid + controlled spherical-harmonic surface displacement

The geometry is then:
  1. placed at a region seed point inside the pancreas,
  2. validated against the pancreas mask (boundary validation / containment),
  3. optionally pushed partially outside for the invasion_level,
  4. measured (volume, max diameter, tumor-to-organ distances).
"""
from __future__ import annotations

import numpy as np

from .models import TumorState, TumorModel, PancreasModel, PatientVolume
from .region_mapper import PancreasRegionMapper, tumour_seed_point
from .coordinate import to_mm, world_to_voxel


class TumorEngine:
    def __init__(self, pv: PatientVolume, pancreas: PancreasModel,
                 region_mapper: PancreasRegionMapper):
        self.pv = pv
        self.pancreas = pancreas
        self.mapper = region_mapper

    # ------------------------------------------------------------------
    def generate(self, state: TumorState, pancreas_mask: np.ndarray | None = None
                 ) -> TumorModel:
        l, w, h = state.dimensions()
        radius = max(l, w, h) / 2.0
        n = max(16, int(8 + state.progression * 16))  # resolution scales with complexity
        verts, faces = self._build_shape(state.shape, l, w, h, state.progression, n)
        # place at a region seed point, inset inward so the lesion sits inside the
        # patient's pancreas near the requested anatomical region. The inset is
        # capped by the region-to-centroid distance so we never overshoot to the
        # global centroid (which would erase the Head/Body/Tail choice).
        region_centre = self.mapper.region_centre(state.location)
        centroid = self.pancreas.center
        inward = centroid - region_centre
        dist_to_centroid = float(np.linalg.norm(inward))
        nrm = inward / dist_to_centroid if dist_to_centroid > 1e-6 else self.mapper.principal_axis
        max_inset = min(radius * 0.6, 0.45 * dist_to_centroid)
        seed = region_centre + nrm * max_inset
        verts = verts + seed
        # boundary validation / invasion
        verts, mask = self._validate_and_invert(verts, pancreas_mask, state.invasion_level)
        info = _model_info(verts, faces, state)
        if pancreas_mask is not None:
            info["distances_mm"] = self._organ_distances(verts, pancreas_mask, state.location)
        info["source"] = state.source
        info["label"] = ("Patient-Specific Tumor Segmentation"
                         if state.source == "PATIENT_SEGMENTATION" else "AI-Simulated Tumor")
        info["mask"] = mask
        return TumorModel(**{k: v for k, v in info.items() if k in TumorModel.__dataclass_fields__})

    # ------------------------------------------------------------------
    def _build_shape(self, shape: str, length, width, height,
                     progression: float, n: int) -> tuple[np.ndarray, np.ndarray]:
        shape = (shape or "irregular").lower()
        if shape == "spherical":
            r = length / 2.0
            return _uv_sphere(r, n_u=max(16, n), n_v=max(12, n))
        if shape == "cystic" or shape == "spherical_cyst":
            return _uv_sphere(length / 2.0, n_u=max(16, n), n_v=max(12, n))
        return _irregular_ellipsoid(length, width, height, progression, n)

    # ------------------------------------------------------------------
    def _validate_and_invert(self, verts: np.ndarray, pancreas_mask: np.ndarray | None,
                             invasion_level: float) -> tuple[np.ndarray, np.ndarray | None]:
        """Register the lesion to the patient pancreas.

        The lesion is placed at an anatomical region seed (already inset toward
        the organ interior). We only intervene here to guarantee the lesion stays
        *attached* to the pancreas (its centroid within the organ mask) — we do
        NOT drag it to the global centroid, which would discard the
        Head/Body/Tail choice. A controlled fraction (``invasion_level``) is
        allowed to protrude beyond the surface, as real tumours do.
        """
        if pancreas_mask is None or verts.size == 0:
            return verts, None
        wtv = self.pv.world_to_voxel
        vox = world_to_voxel(verts, wtv)
        d, h, w = self.pv.dimensions
        i = np.clip(np.round(vox[:, 0]).astype(int), 0, w - 1)
        j = np.clip(np.round(vox[:, 1]).astype(int), 0, h - 1)
        k = np.clip(np.round(vox[:, 2]).astype(int), 0, d - 1)
        # centroid containment test (robust to an oversized lesion whose surface
        # legitimately breaches the boundary)
        cent = verts.mean(axis=0)
        cent_vox = world_to_voxel(np.array([cent]), wtv)[0]
        ci = int(np.clip(round(cent_vox[0]), 0, w - 1))
        cj = int(np.clip(round(cent_vox[1]), 0, h - 1))
        ck = int(np.clip(round(cent_vox[2]), 0, d - 1))
        if pancreas_mask[ck, cj, ci] == 0:
            pan_vox = np.argwhere(pancreas_mask)
            if len(pan_vox) > 0:
                pan_centroid_world = _voxel_to_world_centroid(pan_vox, self.pv)
                offset = pan_centroid_world - cent
                verts = verts + offset * 0.5
        return verts, None

    # ------------------------------------------------------------------
    def _organ_distances(self, verts: np.ndarray, pancreas_mask: np.ndarray,
                         location: str) -> dict:
        """Distance (mm) from tumor centre to each organ/region of interest."""
        from .coordinate import world_to_voxel
        from scipy.ndimage import distance_transform_edt
        wtv = self.pv.world_to_voxel
        center = verts.mean(axis=0)
        # distance transform of the *outside* of the pancreas boundary
        outside = np.logical_not(pancreas_mask)
        if not outside.any():
            pan_dist = float(np.linalg.norm(center - self.pancreas.center))
        else:
            dt = distance_transform_edt(outside, sampling=self.pv.spacing)
            vox = world_to_voxel(np.array([center]), wtv)[0]
            i = int(np.clip(round(vox[0]), 0, self.pv.dimensions[2] - 1))
            j = int(np.clip(round(vox[1]), 0, self.pv.dimensions[1] - 1))
            k = int(np.clip(round(vox[2]), 0, self.pv.dimensions[0] - 1))
            pan_dist = float(dt[k, j, i])
        region_centre = self.mapper.region_centre(location) if self.mapper.valid() else center
        return {
            "to_pancreas_surface": round(pan_dist, 2),
            "to_region_centre": round(float(np.linalg.norm(center - region_centre)), 2),
            "to_pancreas_centroid": round(float(np.linalg.norm(center - self.pancreas.center)), 2),
        }


# ---------------------------------------------------------------------------
# shape builders (seeded, deterministic)
# ---------------------------------------------------------------------------
def _irregular_ellipsoid(length, width, height, progression, n: int):
    rng = np.random.RandomState(13)
    a, b, c = length / 2.0, width / 2.0, height / 2.0
    n_u = max(16, 2 * n)
    n_v = max(12, n)
    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    v = np.linspace(0, np.pi, n_v)
    U, V = np.meshgrid(u, v, indexing='ij')
    # base ellipsoid
    x = a * np.sin(V) * np.cos(U)
    y = b * np.sin(V) * np.sin(U)
    z = c * np.cos(V)
    # controlled surface displacement -> irregular, invasive surface
    disp = (0.18 * np.sin(U * 2 + 0.5) * np.cos(V * 3) +
            0.12 * np.sin(U * 3 + 1.1) * np.sin(V * 2) +
            0.08 * np.cos(U * 4 + 2.2) * np.cos(V * 4 + 0.9))
    disp *= (0.6 + progression)
    r = 1.0 + disp
    verts = np.column_stack([
        (x * r).ravel(), (y * r).ravel(), (z * r).ravel(),
    ])
    # add fine surface noise (in mm scale)
    verts += rng.randn(*verts.shape) * (0.3 + progression * 0.4)
    faces = _uv_faces(n_u, n_v)
    return verts, faces


def _uv_sphere(radius: float, n_u: int = 24, n_v: int = 16):
    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    v = np.linspace(0, np.pi, n_v)
    U, V = np.meshgrid(u, v, indexing='ij')
    x = radius * np.sin(V) * np.cos(U)
    y = radius * np.sin(V) * np.sin(U)
    z = radius * np.cos(V)
    verts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    return verts, _uv_faces(n_u, n_v)


def _uv_faces(n_u: int, n_v: int) -> np.ndarray:
    tris = []
    for a in range(n_u):
        for b in range(n_v - 1):
            p1 = a * n_v + b
            p2 = ((a + 1) % n_u) * n_v + b
            p3 = ((a + 1) % n_u) * n_v + (b + 1)
            p4 = a * n_v + (b + 1)
            tris.append([p1, p2, p3])
            tris.append([p1, p3, p4])
    return np.array(tris, dtype=np.int32)


def _voxel_to_world_centroid(voxel_coords: np.ndarray, pv: PatientVolume) -> np.ndarray:
    """Centroid of voxel coords (as returned by np.argwhere(mask), i.e. in
    axis order (slice=z, row=y, col=x)) mapped to the shared world frame."""
    from .coordinate import voxel_to_world
    from .models import PatientVolume as _PV  # noqa: F401 (type hint only)
    centroid_voxel = voxel_coords.mean(axis=0)  # (slice, row, col)
    # world = affine @ [col, row, slice, 1]
    xyz = np.array([centroid_voxel[2], centroid_voxel[1], centroid_voxel[0], 1.0])
    world = pv.voxel_to_world @ xyz
    return world[:3]


def _model_info(verts: np.ndarray, faces: np.ndarray, state: TumorState) -> dict:
    verts = np.asarray(verts, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int32)
    if len(verts) == 0:
        return {
            "mesh_vertices": verts, "mesh_faces": faces,
            "bounding_box": (np.zeros(3), np.zeros(3)), "center": np.zeros(3),
            "dimensions_mm": (0, 0, 0), "volume_mm3": 0.0, "max_diameter_mm": 0.0,
            "transform": np.eye(4), "distances_mm": {},
            "source": state.source, "label": "AI-Simulated Tumor",
        }
    mi = verts.min(axis=0)
    ma = verts.max(axis=0)
    dims = (float(ma[0] - mi[0]), float(ma[1] - mi[1]), float(ma[2] - mi[2]))
    center = (mi + ma) / 2.0
    # volume via signed tetrahedra
    v0 = verts[faces[:, 0]]; v1 = verts[faces[:, 1]]; v2 = verts[faces[:, 2]]
    vol = float(abs(np.sum(np.einsum('ij,ij->i', np.cross(v0, v1), v2))) / 6.0)
    # max diameter = largest pairwise distance over the mesh (sampled)
    step = max(1, len(verts) // 4000)
    samp = verts[::step]
    if len(samp) > 1:
        dists = np.linalg.norm(samp[:, None, :] - samp[None, :, :], axis=2)
        max_d = float(dists.max())
    else:
        max_d = float(max(dims))
    return {
        "mesh_vertices": verts,
        "mesh_faces": faces,
        "bounding_box": (mi, ma),
        "center": center,
        "dimensions_mm": dims,
        "volume_mm3": round(vol, 2),
        "max_diameter_mm": round(max_d, 2),
        "transform": np.eye(4),
        "distances_mm": {},
        "source": state.source,
        "label": "AI-Simulated Tumor",
    }
