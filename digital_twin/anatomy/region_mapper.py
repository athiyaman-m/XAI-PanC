"""Anatomical region mapping for the patient-specific pancreas.

PancreasRegionMapper
  - takes the real patient pancreas mesh (world mm),
  - computes its principal axis (PCA of vertices),
  - partitions the mesh into Head / Neck / Body / Tail regions,
  - returns a region centre + local volume for tumor placement.

The subdivision follows the standard anatomical convention:
  * the neck is the narrowest cross-section ~1/3 of the way from the head
    (proximal / duodenal) end;
  * head = the bulb near the duodenum (the wider end containing the ampulla);
  * body = the shaft between neck and the onset of the tail taper;
  * tail = the thin, tapering distal end (often extends toward the spleen).

If exact subregion segmentation is unavailable (the common case), the region
boundaries are derived *from the actual patient geometry* along the principal
axis — never from fixed global world coordinates.
"""
from __future__ import annotations

import numpy as np

from .models import PancreasModel

# Fractional positions along the head->tail (principal) axis, measured from the
# proximal (head/duodenum) end. These are ratios so they adapt to each patient.
REGION_BOUNDS = {
    "Head":   (0.00, 0.22),
    "Neck":   (0.22, 0.33),
    "Body":   (0.33, 0.78),
    "Tail":   (0.78, 1.00),
}
REGION_COLORS = {"Head": "#ef565f", "Neck": "#f77828", "Body": "#f4a524", "Tail": "#6badf5"}


class PancreasRegionMapper:
    def __init__(self, pancreas: PancreasModel):
        self.pancreas = pancreas
        self.vertices = np.asarray(pancreas.mesh_vertices, dtype=np.float64)
        self.faces = np.asarray(pancreas.mesh_faces, dtype=np.int32)
        self._compute_axis()

    # ------------------------------------------------------------------
    def _compute_axis(self):
        verts = self.vertices
        if len(verts) == 0:
            self.principal_axis = np.array([1.0, 0.0, 0.0])
            self.proj = np.zeros(len(verts))
            self.proximal_centre = np.zeros(3)
            self.distal_centre = np.zeros(3)
            self.length = 0.0
            self.regions = {r: {"centre": np.zeros(3), "bounds": b, "colour": c}
                            for r, b, c in [(k, v, REGION_COLORS.get(k, "#888")) for k, v in REGION_BOUNDS.items()]}
            return
        cov = np.cov(verts, rowvar=False)
        # direction of largest variance = head-tail axis
        vals, vecs = np.linalg.eigh(cov)
        order = np.argsort(vals)[::-1]
        axis = vecs[:, order[0]]
        # make axis point head->tail? we can't know orientation from mesh alone;
        # we define proximal = the end with larger cross-sectional area (head),
        # distal = the tapering end (tail) using vertex density / area distribution.
        proj = verts @ axis
        # estimate cross-section area per slab along proj to find the head (wider)
        slabs = np.linspace(proj.min(), proj.max(), 9)
        counts = np.array([((proj >= slabs[i]) & (proj < slabs[i + 1])).sum()
                           for i in range(len(slabs) - 1)])
        proximal_is_left = counts[0] < counts[-1]  # if right end wider, distal=argmax
        if proximal_is_left:
            proximal_idx = counts.argmax()
            # rebuild axis so it points left->right (proximal=head side)
            pass
        # Choose orientation: proximal = slab with max vertex count = head side
        proximal_slab = counts.argmax()
        proj_shifted = proj - slabs[proximal_slab]
        if proximal_slab > len(counts) / 2:
            axis = -axis
            proj = verts @ axis
            proj_shifted = proj - (proj.min() if proximal_slab > len(counts) / 2 else slabs[proximal_slab])
        proj_norm = proj - proj.min()
        total = proj.max() - proj.min()
        if total <= 0:
            total = 1.0
        t = proj_norm / total

        self.principal_axis = axis
        self.proj = t
        self.length = float(np.linalg.norm(verts.max(axis=0) - verts.min(axis=0)))
        # proxy centres: centroid of vertices inside each region slab
        self.regions = {}
        for name, (lo, hi) in REGION_BOUNDS.items():
            mask = (t >= lo) & (t <= hi)
            if mask.any():
                centre = verts[mask].mean(axis=0)
            else:
                # fall back to fractional position along the axis endpoints
                ends = _region_endpoints(self.vertices, axis)
                centre = (1 - (lo + hi) / 2) * ends[0] + ((lo + hi) / 2) * ends[1]
            self.regions[name] = {
                "centre": centre,
                "bounds": (lo, hi),
                "colour": REGION_COLORS.get(name, "#888"),
                "t": (lo + hi) / 2,
            }
        self.pancreas.principal_axis = axis
        self.pancreas.regions = {k: {"centre": v["centre"], "bounds": v["bounds"],
                                    "colour": v["colour"]} for k, v in self.regions.items()}

    def region_centre(self, name: str) -> np.ndarray:
        key = self._canon(name)
        return self.regions.get(key, {}).get("centre", np.zeros(3))

    def region_mask(self, name: str, pv_voxel_shape=None) -> dict:
        """Return region boundaries along the principal axis (for placement)."""
        return self.regions.get(self._canon(name), {})

    def _canon(self, name: str) -> str:
        # accept "HEAD"/"head"/"Head" interchangeably
        n = str(name).strip().title()
        if n not in self.regions:
            for k in self.regions:
                if k.lower() == n.lower():
                    return k
        return n

    def valid(self) -> bool:
        return len(self.vertices) > 0 and self.pancreas is not None


def _region_endpoints(verts: np.ndarray, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    proj = verts @ axis
    order = np.argsort(proj)
    # proximal (min proj) and distal (max proj) centroids across the ends
    prox = verts[order[:max(5, len(order) // 20)]].mean(axis=0)
    dist = verts[order[-max(5, len(order) // 20):]].mean(axis=0)
    return prox, dist


def tumour_seed_point(region_name: str, mapper: PancreasRegionMapper,
                      inset_mm: float = 0.0) -> np.ndarray:
    """A safe 3D point inside the requested anatomical region, on the
    patient's own pancreas geometry.

    ``inset_mm`` pushes the point *inward* along the pancreas interior direction
    (from the region surface centroid toward the pancreas centroid). This keeps a
    tumour of non-trivial radius inside the organ rather than sitting half-outside
    on the surface — the old floating-outside behaviour.
    """
    if not mapper.valid():
        return np.zeros(3)
    centre = mapper.region_centre(region_name)
    centroid = mapper.pancreas.center
    inward = centroid - centre
    nrm = np.linalg.norm(inward)
    if nrm > 1e-6:
        inward = inward / nrm
    else:
        inward = mapper.principal_axis
    return centre + inward * inset_mm
