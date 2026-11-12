import os
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from skimage.measure import marching_cubes
except Exception:  # pragma: no cover - optional dependency
    marching_cubes = None

try:
    import pydicom
except Exception:  # pragma: no cover - optional dependency
    pydicom = None


def discover_ct_candidates(root: str | os.PathLike[str] | None = None) -> list[Path]:
    base = Path(root or os.getcwd())
    patterns = ["**/*.dcm", "**/*.DCM", "**/*.nii", "**/*.nii.gz", "**/*.npy", "**/*.npz"]
    candidates: list[Path] = []
    for pattern in patterns:
        candidates.extend(base.glob(pattern))
    return sorted({p.resolve() for p in candidates if p.exists()})


def _load_dicom_volume(path: Path) -> Optional[np.ndarray]:
    if pydicom is None:
        return None
    try:
        ds = pydicom.dcmread(str(path), force=False)
    except Exception:
        return None

    pixel_array = getattr(ds, "pixel_array", None)
    if pixel_array is None:
        return None

    array = np.asarray(pixel_array, dtype=np.float32)
    if array.ndim != 3:
        return None
    return array


def _load_npy_volume(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix.lower() == ".npy":
            return np.load(path, allow_pickle=False)
        if path.suffix.lower() == ".npz":
            data = np.load(path, allow_pickle=False)
            return np.asarray(next(iter(data.values())), dtype=np.float32)
    except Exception:
        return None
    return None


def load_real_ct_volume(candidate_paths: Optional[list[Path]] = None) -> Optional[dict]:
    search_roots = []
    if candidate_paths:
        search_roots = candidate_paths
    else:
        for base in [
            Path("DATASETS"),
            Path("backend/data/uploads"),
            Path("data"),
            Path("./DATASETS"),
        ]:
            if base.exists():
                search_roots.append(base)

    if not search_roots:
        return None

    for root in search_roots:
        candidates = discover_ct_candidates(root)
        for path in candidates:
            if path.suffix.lower() in {".dcm", ".dicom", ".DCM"}:
                volume = _load_dicom_volume(path)
            elif path.suffix.lower() in {".npy", ".npz"}:
                volume = _load_npy_volume(path)
            elif path.suffix.lower() in {".nii", ".nii.gz"}:
                continue
            else:
                continue
            if volume is not None and np.isfinite(volume).all():
                return {
                    "volume": volume,
                    "spacing": (1.0, 1.0, 1.0),
                    "source": "ct",
                    "path": str(path),
                }
    return None


def create_pancreas_surface_from_volume(
    volume: np.ndarray,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    threshold: Optional[float] = None,
) -> dict:
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError("volume must be a 3D array")
    if marching_cubes is None:
        raise ImportError("scikit-image is required for marching cubes meshing")

    if threshold is None:
        threshold = float(np.percentile(arr, 80))
    if not np.isfinite(threshold):
        threshold = 0.5

    verts, faces, _, _ = marching_cubes(arr, level=threshold, step_size=1)
    if len(verts) == 0 or len(faces) == 0:
        raise ValueError("could not extract a valid surface from volume")

    verts = verts * np.array(spacing, dtype=np.float64)
    return {
        "vertices": verts.astype(np.float64),
        "faces": faces.astype(np.int32),
        "source": "ct",
        "path": None,
    }


def build_tumor_pancreas_mesh(
    volume_shape: tuple[int, int, int] = (64, 64, 64),
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    tumor_size: float = 0.24,
    tumor_strength: float = 0.35,
    tumor_offset: tuple[float, float, float] = (0.0, 0.10, 0.15),
) -> dict:
    shape = tuple(volume_shape)
    if len(shape) != 3:
        raise ValueError("volume_shape must be length 3")

    x = np.linspace(-1.0, 1.0, shape[0])
    y = np.linspace(-1.0, 1.0, shape[1])
    z = np.linspace(-1.0, 1.0, shape[2])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Pancreas-like anatomical envelope: elongated, tapered and slightly curved.
    body = np.exp(-((X * 1.3) ** 2 + (Y * 1.1) ** 2 + (Z * 1.0) ** 2) / 0.42)
    taper = np.exp(-((X + 0.25) ** 2) / 0.20) * 0.5 + np.exp(-((X - 0.25) ** 2) / 0.20) * 0.4
    body = np.clip(body + 0.2 * taper, 0.0, 1.0)

    # Tumor inclusion as a focal high-intensity nodule.
    ox, oy, oz = tumor_offset
    tumor_mask = np.exp(-(((X - ox) ** 2) / (max(tumor_size, 0.05) ** 2) + ((Y - oy) ** 2) / (max(tumor_size, 0.05) ** 2) + ((Z - oz) ** 2) / (max(tumor_size, 0.05) ** 2)))
    volume = np.clip(body + tumor_strength * tumor_mask, 0.0, 1.0).astype(np.float32)

    if marching_cubes is None:
        raise ImportError("scikit-image is required for marching cubes meshing")

    verts, faces, _, _ = marching_cubes(volume, level=0.56, step_size=1)
    if len(verts) == 0 or len(faces) == 0:
        raise ValueError("could not build a valid pancreas mesh")

    verts = verts * np.array(spacing, dtype=np.float64)
    return {
        "vertices": verts.astype(np.float64),
        "faces": faces.astype(np.int32),
        "source": "prebuilt",
        "path": None,
    }


def load_pancreas_mesh_from_ct_or_fallback() -> dict:
    ct_data = load_real_ct_volume()
    if ct_data is not None:
        try:
            mesh = create_pancreas_surface_from_volume(
                ct_data["volume"],
                spacing=ct_data.get("spacing", (1.0, 1.0, 1.0)),
                threshold=None,
            )
            mesh["source"] = "ct"
            mesh["path"] = ct_data.get("path")
            return mesh
        except Exception:
            pass

    return build_tumor_pancreas_mesh(
        volume_shape=(64, 64, 64),
        spacing=(1.0, 1.0, 1.0),
        tumor_size=0.24,
        tumor_strength=0.40,
    )
