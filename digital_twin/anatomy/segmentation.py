"""Segmentation backend for the CT-based digital twin.

Interface
---------
class SegmentationBackend:
    segment_pancreas(pv: PatientVolume, label_path=None) -> np.ndarray
    segment_tumor(pv, pancreas_mask, label_path=None) -> np.ndarray | None

The pancreas mask stays registered to the PatientVolume voxel grid (the same
frame the mesh is extracted from). Two realisation modes:

  * REAL annotation / segmentation label (NIfTI) provided -> loaded directly
    (MODE 2: "Patient-Specific Tumor Segmentation").
  * No annotation -> a learned model when available (TotalSegmentator's
    ``pancreas`` task, or a MONAI model) -> else a documented atlas/HU
    fallback (``source="atlas_fallback"``) derived from the actual CT voxels.

The atlas fallback is an *approximation* produced entirely from the patient's
own CT (HU tissue window + anatomical spatial prior), so the pipeline runs end
to end without a dedicated model. When TotalSegmentator is installed the
pipeline transparently uses it; no code change is required at call sites.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from .models import PatientVolume
from .coordinate import voxel_to_world

log = logging.getLogger("ExAI-PanC.anatomy.segmentation")

# Try optional learned segmenters at import time.
_HAS_TS = False
try:  # pragma: no cover - env dependent
    from totalsegmentator.python_interface import TotalSegmentator
    _HAS_TS = True
except Exception:
    try:
        from totalsegmentator import TotalSegmentator
        _HAS_TS = True
    except Exception:
        _HAS_TS = False

_HAS_MONAI = False
try:  # pragma: no cover - env dependent
    import monai  # noqa: F401
    _HAS_MONAI = True
except Exception:
    _HAS_MONAI = False


class SegmentationBackend:
    """Segment pancreas (and optionally tumor) from a registered PatientVolume."""

    def __init__(self, tmp_dir: str = "models/_seg_cache"):
        self.tmp_dir = Path(tmp_dir)
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.engine = "fallback"

    # ------------------------------------------------------------------
    # Pancreas
    # ------------------------------------------------------------------
    def segment_pancreas(self, pv: PatientVolume, label_path: Optional[str] = None
                         ) -> tuple[np.ndarray, dict]:
        if label_path and Path(label_path).exists():
            mask, info = self._load_label(label_path, pv)
            info["source"] = "labelmap"
            info["backend"] = "ground_truth"
            return mask.astype(np.uint8), info
        if _HAS_TS:
            return self._segment_totalsegmentator(pv, "pancreas")
        if _HAS_MONAI:
            # MONAI segmentation hook (delegated; requires a downloaded bundle).
            mask = self._segment_monai(pv, "pancreas")
            if mask is not None:
                return mask.astype(np.uint8), {"source": "monai", "backend": "monai"}
        mask, info = self._atlas_pancreas(pv)
        return mask.astype(np.uint8), info

    # ------------------------------------------------------------------
    # Tumor
    # ------------------------------------------------------------------
    def segment_tumor(self, pv: PatientVolume, pancreas_mask: np.ndarray,
                      label_path: Optional[str] = None) -> Optional[np.ndarray]:
        """Return a tumor mask registered to the volume, or None if unavailable."""
        if label_path and Path(label_path).exists():
            mask, _ = self._load_label(label_path, pv)
            return mask.astype(np.uint8)
        if _HAS_TS:
            try:
                mask, _ = self._segment_totalsegmentator(pv, "pancreaticcancer")
                return mask.astype(np.uint8)
            except Exception:
                pass
        return None

    # ------------------------------------------------------------------
    # Learned backends
    # ------------------------------------------------------------------
    def _segment_totalsegmentator(self, pv: PatientVolume, task: str):
        self.engine = "totalsegmentator"
        inp = pv.source_path
        out_dir = self.tmp_dir / f"ts_{Path(inp).stem}"
        out_dir.mkdir(parents=True, exist_ok=True)
        # TotalSegmentator works in LPS world coordinates natively (it handles
        # DICOM/NIfTI), then we re-register the resulting mask to our voxel grid.
        TotalSegmentator(input=inp, output=str(out_dir),
                         task=task, ml=True, fast=False)
        label = _load_totalsegmentator_output(out_dir, task)
        if label is None:
            raise RuntimeError("TotalSegmentator produced no mask")
        # resample label onto pv voxel grid (same RAS frame) -> already aligned for NIfTI inputs
        mask = _resample_label_to_grid(label, pv).astype(np.uint8)
        info = {"source": "totalsegmentator", "backend": "totalsegmentator", "task": task}
        return mask, info

    def _segment_monai(self, pv, task: str):
        # Hook for a MONAI bundle; not wired without a model artifact.
        return None

    # ------------------------------------------------------------------
    # Atlas / HU fallback
    # ------------------------------------------------------------------
    def _atlas_pancreas(self, pv: PatientVolume) -> tuple[np.ndarray, dict]:
        """Atlas/HU fallback: pancreas tissue + anatomical spatial prior from the
        patient's own CT. Clearly an approximation; documented as fallback.
        """
        vol = np.asarray(pv.image, dtype=np.float32)
        d, h, w = vol.shape

        # 1) HU tissue window for pancreas (contrast-enhanced: soft tissue ~ -50..150).
        tissue = (vol >= -55) & (vol <= 140)

        # 2) anatomical spatial prior: pancreas lives in the upper abdomen, around
        #    the SMA/SMV confluence. Built directly in patient physical (RAS, mm)
        #    coords so it is tied to the patient's own anatomy — never a fixed
        #    global coordinate.
        vmin, vmax = pv.world_box()
        centre_world = (vmin + vmax) / 2.0
        half = np.array([120.0, 70.0, 140.0])          # mm, generous upper-abdomen box
        box_centre_world = centre_world + np.array([0.0, -30.0, 0.0])
        zs, ys, xs = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing='ij')
        vox = np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1).astype(np.float64)
        world = voxel_to_world(vox, pv.voxel_to_world).reshape(d, h, w, 3)
        prior_mask = (np.abs(world - box_centre_world) <= half).all(axis=-1)

        combined = tissue & prior_mask

        # 3) morphological refinement — keep the largest connected component
        combined = _largest_component(combined)
        combined = _smooth_mask(combined, iterations=2)

        info = {
            "source": "atlas_fallback",
            "backend": "hu_atlas",
            "method": "HU tissue window (-55..140 HU) & anatomical spatial prior",
            "note": "approximation; install TotalSegmentator for production",
            "voxel_count": int(combined.sum()),
        }
        return combined.astype(np.uint8), info

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _load_label(self, label_path: str, pv: PatientVolume) -> tuple[np.ndarray, dict]:
        try:
            import nibabel as nib
            img = nib.load(str(label_path))
            label = np.asarray(img.get_fdata(), dtype=np.uint8)
            aff = np.asarray(img.affine, dtype=np.float64)
            label, aff = _ensure_ras_mask(label, aff)
            mask = _resample_label_to_grid(label, pv, src_affine=aff)
            info = {"source": "labelmap", "backend": "ground_truth",
                    "label_path": str(label_path)}
            return mask.astype(np.uint8), info
        except Exception as e:
            log.warning("failed to load label %s: %s", label_path, e)
            raise

    @staticmethod
    def is_real_segmenter_available() -> bool:
        return _HAS_TS or _HAS_MONAI


def _ensure_ras_mask(mask: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    aff = np.asarray(affine, dtype=np.float64)
    flips = [aff[i, i] < 0 for i in range(3)]
    if any(flips):
        mask = np.flip(mask, axis=tuple(ax for ax, f in enumerate(flips) if f)).copy()
        new_aff = aff.copy()
        for ax in range(3):
            if flips[ax]:
                new_aff[ax, ax] = abs(new_aff[ax, ax])
                new_aff[ax, 3] = aff[ax, 3] + (mask.shape[ax] - 1) * abs(aff[ax, ax])
        new_aff[3, 3] = 1.0
        return mask, new_aff
    return mask, aff


def _load_totalsegmentator_output(out_dir: Path, task: str):
    """Find the segmentation probability/label file written by TotalSegmentator."""
    cands = []
    for p in out_dir.rglob("*"):
        if p.suffix.lower() in (".nii", ".nii.gz", ".mhd", ".nii.gz"):
            cands.append(p)
    if not cands:
        return None
    try:
        import nibabel as nib
        img = nib.load(str(cands[0]))
        return np.asarray(img.get_fdata(), dtype=np.float32), np.asarray(img.affine, dtype=np.float64)
    except Exception:
        return None


def _resample_label_to_grid(label, pv: PatientVolume, src_affine=None) -> np.ndarray:
    """Resample a labelmap (with src_affine) onto pv's voxel grid (nearest)."""
    try:
        from scipy.ndimage import map_coordinates
    except Exception:  # pragma: no cover
        return label
    if src_affine is None:
        src_affine = pv.voxel_to_world
    target = label.astype(np.float32)
    # map each target voxel index to source voxel coords
    d, h, w = pv.dimensions
    zs, ys, xs = np.meshgrid(np.arange(d), np.arange(h), np.arange(w), indexing='ij')
    target_world = voxel_to_world(np.stack([xs.ravel(), ys.ravel(), zs.ravel()], axis=1),
                                  pv.voxel_to_world)
    wtv = np.linalg.inv(src_affine)
    src_idx = target_world @ wtv.T
    src_idx = src_idx[:, :3]  # (N,3)
    # source indices are (i=x, j=y, k=z) in nibabel orientation (x,y,z)
    i = src_idx[:, 0]
    j = src_idx[:, 1]
    k = src_idx[:, 2]
    coords = [k.reshape(label.shape), j.reshape(label.shape), i.reshape(label.shape)]
    out = np.zeros_like(target)
    out = map_coordinates(target, coords, order=0, mode='constant', cval=0.0)
    return (out > 0.5).astype(np.uint8)


def _upsample_label(label: np.ndarray, target_shape: tuple[int, int, int]) -> np.ndarray:
    """Nearest-neighbour upsample a small label to target_shape."""
    try:
        from scipy.ndimage import zoom
    except Exception:  # pragma: no cover
        return label
    if label.shape == target_shape:
        return label
    factors = [t / s for t, s in zip(target_shape, label.shape)]
    if any(f <= 0 for f in factors):
        return np.zeros(target_shape, dtype=label.dtype)
    out = zoom(label.astype(np.float32), factors, order=0, mode='nearest')
    return (out > 0.5).astype(label.dtype)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    try:
        from scipy.ndimage import label as ndi_label
        labeled, n = ndi_label(mask)
        if n == 0:
            return mask
        sizes = np.bincount(labeled.ravel())[1:]
        biggest = int(np.argmax(sizes)) + 1
        return (labeled == biggest)
    except Exception:  # pragma: no cover
        return mask


def _smooth_mask(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    try:
        from scipy.ndimage import binary_opening, binary_closing
        out = mask
        for _ in range(iterations):
            out = binary_opening(out, structure=np.ones((3, 3, 3)))
            out = binary_closing(out, structure=np.ones((3, 3, 3)))
        return out
    except Exception:  # pragma: no cover
        return mask
