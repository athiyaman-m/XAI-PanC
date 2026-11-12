"""Export a patient CT volume as a browser-ready 3D volume-rendering payload.

The Digital Twin frontend volume-renders the patient's real CT (the actual
DICOM/NIfTI voxels, not just the segmented meshes).  This module converts a
reconstructed HU volume into a small windowed uint8 grid (raw-deflate + base64)
plus the physical spacing and an optional tumour anchor, so the React/R3F
renderer can highlight the tumour and scale a live biomarker-driven impact
region around it.
"""
from __future__ import annotations

import base64
import zlib
from typing import Any, Iterable

import numpy as np

# Soft-tissue CT window used for the web volume (HU).
HU_WINDOW = (-160.0, 240.0)
# Browser-friendly grid budget: keep axial resolution near 256 and ≤ 96 slices.
MAX_AXIAL = 256
MAX_SLICES = 96
ZLIB_LEVEL = 6


def downsample_volume(volume: np.ndarray, spacing: tuple[float, float, float]):
    """Stride the volume to a web-friendly grid.

    Returns ``(volume_downsampled, spacing_downsampled, strides)`` where the
    strides let callers map full-resolution voxel coordinates (e.g. a tumour
    anchor) onto the downsampled grid.
    """
    z, y, x = volume.shape
    sz = max(1, int(np.ceil(z / MAX_SLICES)))
    sy = max(1, int(np.ceil(y / MAX_AXIAL)))
    sx = max(1, int(np.ceil(x / MAX_AXIAL)))
    vol = np.ascontiguousarray(volume[::sz, ::sy, ::sx])
    spacing_d = (spacing[0] * sz, spacing[1] * sy, spacing[2] * sx)
    return vol, spacing_d, (sz, sy, sx)


def window_to_uint8(volume: np.ndarray, wmin: float, wmax: float) -> np.ndarray:
    """Map HU values to 0..255 inside the requested window (0 = transparent)."""
    span = max(1e-6, float(wmax - wmin))
    v = np.clip((np.asarray(volume, dtype=np.float32) - wmin) / span, 0.0, 1.0)
    return (v * 255.0).astype(np.uint8)


def _encode_raw_deflate(data: np.ndarray) -> str:
    compressor = zlib.compressobj(level=ZLIB_LEVEL, wbits=-zlib.MAX_WBITS)
    raw = compressor.compress(data.tobytes()) + compressor.flush()
    return base64.b64encode(raw).decode("ascii")


def anchor_from_mask(mask: np.ndarray, strides: tuple[int, int, int]) -> list[int] | None:
    """Centroid voxel of a (z, y, x) label mask on the downsampled grid."""
    if mask is None:
        return None
    m = mask[:: strides[0], :: strides[1], :: strides[2]]
    idx = np.argwhere(np.asarray(m) > 0)
    if len(idx) == 0:
        return None
    return [int(v) for v in np.clip(np.round(idx.mean(axis=0)), 0, np.array(m.shape) - 1)]


def build_ct_volume_payload(
    volume_hu: np.ndarray,
    spacing: tuple[float, float, float],
    title: str,
    source: str,
    tumor_voxel: Iterable[int] | None = None,
    tumor_cm3: float | None = None,
    tumor_source: str = "default",
    tumor_label: str | None = None,
    window: tuple[float, float] = HU_WINDOW,
) -> dict[str, Any]:
    """Build the `{width,height,depth,spacing_mm,window,data_b64,tumor}` payload."""
    volume_hu = np.asarray(volume_hu, dtype=np.float32)
    if volume_hu.ndim != 3:
        raise ValueError("volume_hu must be 3D (z, y, x) HU values")
    vol, spacing_d, strides = downsample_volume(volume_hu, spacing)
    d, h, w = vol.shape
    u8 = window_to_uint8(vol, *window)

    voxel = None
    if tumor_voxel is not None:
        voxel = [int(v // s) for v, s in zip(tumor_voxel, strides)]

    raw_bytes = u8.tobytes()
    encoded = _encode_raw_deflate(u8)
    payload: dict[str, Any] = {
        "mode": "ct-volume",
        "title": title,
        "width": int(w),
        "height": int(h),
        "depth": int(d),
        "spacing_mm": [float(spacing_d[0]), float(spacing_d[1]), float(spacing_d[2])],
        "window": [float(window[0]), float(window[1])],
        "data_b64": encoded,
        "tumor": {
            "source": tumor_source,
            "label": tumor_label,
            "cm3": tumor_cm3,
            "voxel": voxel,
        },
        "meta": {
            "source": source,
            "stride": list(strides),
            "voxels": int(u8.size),
            "raw_bytes": len(raw_bytes),
            "encoded_bytes": len(encoded),
        },
    }
    return payload
