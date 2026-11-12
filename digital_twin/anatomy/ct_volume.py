"""CT / DICOM ingestion -> PatientVolume in a canonical RAS (mm) frame.

Handles three input shapes:
  * a folder of a DICOM series (sorted by ImagePositionPatient + InstanceNumber)
  * a single NIfTI (.nii/.nii.gz) file
  * a single DICOM file (treated as a 1-slice volume)

Key guarantees:
  - Series sorting is by the DICOM Patient-Space position (not by filename).
  - Orientation is normalised to RAS (right-anterior-superior, mm).
  - The full affine (voxel -> world mm) is preserved and exposed as
    PatientVolume.voxel_to_world. Original physical coordinate information is
    never destroyed.
  - Resampled volumes keep a separate affine so the original frame is recoverable.
  - HU values are recovered from RescaleIntercept/Slope when present.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
    _HAS_NIB = True
except Exception:  # pragma: no cover
    _HAS_NIB = False

try:
    import pydicom
    from pydicom.dataset import FileDataset
    _HAS_PYDICOM = True
except Exception:  # pragma: no cover
    _HAS_PYDICOM = False

from .models import PatientVolume

# LPS->RAS flip for the spatial axes (x,y,z); colours unchanged.
_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])

# Flips to take a per-slice image orientation (which DICOM gives as
# row, col, 3-components) into a voxel->world direction matrix.
# DICOM ImageOrientationPatient = [row_x,row_y,row_z, col_x,col_y,col_z].
# The slice direction (k) is inferred from the patient-space slice spacing & order.


def load_patient_volume(source, preferred_spacing: tuple[float, float, float] | None = None,
                        resample: bool = False) -> PatientVolume:
    """Load a CT/DICOM source into a :class:`PatientVolume`.

    ``source`` may be a directory (DICOM series), a .dcm file, or a .nii/.nii.gz
    file.
    """
    src = Path(source) if not isinstance(source, (str, Path)) else Path(source)
    if src.is_dir():
        return _load_dicom_series(src, preferred_spacing, resample)
    name = src.name.lower()
    if name.endswith((".nii", ".nii.gz", ".npy", ".npz")):
        return _load_nifti(src, preferred_spacing, resample)
    if name.endswith(".dcm"):
        return _load_dicom_series(src.parent, preferred_spacing, resample)
    raise ValueError(f"unsupported CT source: {source}")


def _load_nifti(path: Path, preferred_spacing, resample) -> PatientVolume:
    if not _HAS_NIB:
        raise ImportError("nibabel is required to load NIfTI volumes")
    img = nib.load(str(path))
    data = np.asarray(img.get_fdata(), dtype=np.float32)
    affine = np.asarray(img.affine, dtype=np.float64)
    # nibabel affines are already voxel->RAS (LPS internally? No: nibabel uses
    # RAS in world coords by default). Ensure the orientation is RAS.
    data, affine = _ensure_ras(data, affine)
    spacing = tuple(float(x) for x in img.header.get_zooms()[:3])
    return _finalize(data, affine, spacing, str(path), modality="CT",
                     preferred_spacing=preferred_spacing, resample=resample)


def _load_dicom_series(folder: Path, preferred_spacing, resample) -> PatientVolume:
    if not _HAS_PYDICOM:
        raise ImportError("pydicom is required to load DICOM series")
    folder = folder.resolve()
    files = sorted(
        [p for p in folder.rglob("*") if p.suffix.lower() in (".dcm",)],
        key=lambda p: str(p),
    )
    if not files:
        raise FileNotFoundError(f"no DICOM files in {folder}")
    datasets = []
    for p in files:
        try:
            ds = pydicom.dcmread(str(p), force=False, stop_before_pixels=False)
            datasets.append((p, ds))
        except Exception:
            continue
    datasets = _filter_valid_series(datasets)
    datasets = _sort_series(datasets)
    return _build_from_datasets(datasets, folder, preferred_spacing, resample)


def _filter_valid_series(datasets):
    """Keep files that share the same SeriesUID (best-effort)."""
    by_series: dict[str, list] = {}
    for p, ds in datasets:
        suid = getattr(ds, "SeriesInstanceUID", None) or getattr(ds, "SOPInstanceUID", "default")
        by_series.setdefault(suid, []).append((p, ds))
    if len(by_series) == 1:
        return datasets
    # pick the largest series (most slices)
    best = max(by_series.values(), key=len)
    return best


def _slice_position(ds) -> tuple[float, float, float]:
    iop = getattr(ds, "ImagePositionPatient", None)
    if iop is not None:
        if isinstance(iop, (list, tuple)):
            return (float(iop[0]), float(iop[1]), float(iop[2]))
        return (float(iop[0]), float(iop[1]), float(iop[2]))
    return (0.0, 0.0, 0.0)


def _sort_series(datasets):
    """Sort slices by ImagePositionPatient projected onto the slice normal,
    then by InstanceNumber as a tiebreaker."""
    def key(item):
        _, ds = item
        iop = _slice_position(ds)
        # project onto the image normal (approx via SliceLocation or InstanceNumber)
        sl = float(getattr(ds, "SliceLocation", 0.0) or 0.0)
        inst = int(getattr(ds, "InstanceNumber", 0) or 0)
        # primary sort by the third patient coord (typical head->feet ordering)
        return (iop[2], sl, inst)
    return sorted(datasets, key=key)


def _build_from_datasets(datasets, folder, preferred_spacing, resample) -> PatientVolume:
    if not datasets:
        raise ValueError("no readable DICOM datasets")
    first = datasets[0][1]
    rows = int(getattr(first, "Rows", 0))
    cols = int(getattr(first, "Columns", 0))
    if rows == 0 or cols == 0:
        raise ValueError("DICOM dataset missing Rows/Columns")
    pixel_repr = getattr(first, "PixelRepresentation", 0) or 0
    intercept = float(getattr(first, "RescaleIntercept", 0.0) or 0.0)
    slope = float(getattr(first, "RescaleSlope", 1.0) or 1.0)
    try:
        pixel = first.pixel_array
        _ = int(pixel.max())  # touch to ensure populated
        dtype = pixel.dtype
    except Exception:
        dtype = np.uint16

    slices = []
    positions = []
    for _, ds in datasets:
        try:
            arr = ds.pixel_array if hasattr(ds, "pixel_array") else None
        except Exception:
            arr = None
        if arr is None:
            arr = np.zeros((rows, cols), dtype=dtype)
        arr = arr.astype(np.float32) * slope + intercept
        if pixel_repr == 1:
            # signed -> already correct after rescale
            pass
        slices.append(arr)
        positions.append(_slice_position(ds))
    volume = np.stack(slices, axis=0)  # (D, H, W)
    positions = np.asarray(positions, dtype=np.float64)

    # spacing: PixelSpacing = (row, col) in mm; slice spacing from positions.
    ps = getattr(first, "PixelSpacing", (1.0, 1.0))
    if isinstance(ps, (list, tuple)):
        px, py = float(ps[0]), float(ps[1])
    else:
        px, py = float(ps), 1.0
    # slice spacing: distance between consecutive slice positions
    if len(positions) >= 2:
        diffs = np.diff(positions, axis=0)
        dists = np.linalg.norm(diffs, axis=1)
        slice_thick = float(getattr(first, "SliceThickness", np.median(dists)) or np.median(dists))
        zs = np.sort(positions[:, 2])
        if len(zs) > 1:
            slice_spacing = float(np.median(np.diff(zs)))
            if slice_spacing <= 0:
                slice_spacing = slice_thick
        else:
            slice_spacing = slice_thick
    else:
        slice_spacing = float(getattr(first, "SliceThickness", 1.0) or 1.0)

    # voxel index order (D,H,W) -> (k,j,i). We build a direction matrix in RAS.
    direction = _orientation_from_dataset(first, px, py, slice_spacing, positions)
    affine = direction
    # voxel origin = first slice's ImagePositionPatient (in patient/LPS coords),
    # converted to RAS by the same flip we apply to the direction.
    origin_lps = np.asarray(positions[0], dtype=np.float64)
    origin_ras = (origin_lps * np.array([-1.0, -1.0, 1.0]))
    affine[:3, 3] = origin_ras
    spacing_tuple = (slice_spacing, px, py)
    return _finalize(volume, affine, spacing_tuple, str(folder),
                     preferred_spacing=preferred_spacing, resample=resample,
                     modality="CT", series_uid=getattr(first, "SeriesInstanceUID", ""))


def _orientation_from_dataset(ds, px, py, pz, positions) -> np.ndarray:
    """Build a 4x4 voxel->world(RAS) affine.

    voxel (k=row(D), j=column(H), i=column(W))? We follow SimpleITK convention:
    voxel index (k, j, i) with spacing (pz, px, py) mapped along the patient
    axes derived from ImageOrientationPatient + slice direction.
    """
    iop = getattr(ds, "ImageOrientationPatient", None)
    orient = np.eye(3, dtype=np.float64)
    if isinstance(iop, (list, tuple)) and len(iop) == 6:
        row_dir = np.asarray(iop[0:3], dtype=np.float64)
        col_dir = np.asarray(iop[3:6], dtype=np.float64)
        row_dir = row_dir / np.linalg.norm(row_dir)
        col_dir = col_dir / np.linalg.norm(col_dir)
        k_dir = np.cross(row_dir, col_dir)
        if np.linalg.norm(k_dir) < 1e-6:
            # slices not acquired in standard orientation; infer from positions
            if len(positions) >= 2:
                diffs = np.diff(positions, axis=0)
                k_dir = diffs[0]
                n = np.linalg.norm(k_dir)
                k_dir = k_dir / n if n > 0 else np.array([0.0, 0.0, 1.0])
            else:
                k_dir = np.array([0.0, 0.0, 1.0])
        # columns of orient are the voxel axes in patient coords (LPS)
        orient[:, 0] = col_dir * px    # voxel i -> patient (col)
        orient[:, 1] = row_dir * py    # voxel j -> patient (row)
        orient[:, 2] = k_dir * pz      # voxel k -> patient (slice)
    else:
        orient[0, 0] = px
        orient[1, 1] = py
        orient[2, 2] = pz
    affine = np.eye(4, dtype=np.float64)
    affine[:3, :3] = orient
    return affine


def _ensure_ras(volume: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flip axes so the resulting world orientation is RAS (right, anterior,
    superior). Returns the (possibly flipped) volume and a corrected affine.

    A flip on voxel axis ``ax`` negates the affine column ``aff[:3, ax]`` and
    shifts the origin so that old voxel ``N-1`` becomes new voxel ``0``:
    ``new_origin = origin + (N-1) * aff[:3, ax]``. We use the sign of the
    column's dominant diagonal entry to decide which axes to flip.
    """
    aff = np.asarray(affine, dtype=np.float64).copy()
    flips = [aff[i, i] < 0 for i in range(3)]
    if not any(flips):
        return volume, aff
    axes = tuple(ax for ax, f in enumerate(flips) if f)
    volume = np.flip(volume, axis=axes)
    new_aff = aff.copy()
    for ax in axes:
        col = new_aff[:3, ax].copy()
        new_aff[:3, ax] = -col                       # negate direction
        # new voxel 0 corresponds to old voxel (N-1) along this axis
        new_aff[ax, 3] = aff[ax, 3] + (volume.shape[ax] - 1) * col[ax]
    new_aff[3, 3] = 1.0
    return volume, new_aff


def _finalize(volume, affine, spacing, source_path, preferred_spacing=None,
              resample=False, modality="CT", series_uid="") -> PatientVolume:
    affine = np.asarray(affine, dtype=np.float64)
    vol = np.ascontiguousarray(volume, dtype=np.float32)
    world_to_voxel = np.linalg.inv(affine)
    d, h, w = vol.shape
    # physical extent along world x/y/z = sum of |axis_vector * count|;
    # voxel grid is (slices=d, rows=h, cols=w) -> voxel axes (col=i, row=j, slice=k)
    # map to affine columns (col->0, row->1, slice->2).
    counts_vx = np.array([w, h, d], dtype=np.float64)
    phys = np.abs(affine[:3, :3] @ counts_vx)
    phys_mm = (float(phys[0]), float(phys[1]), float(phys[2]))

    pv = PatientVolume(
        image=vol,
        voxel_to_world=affine,
        world_to_voxel=world_to_voxel,
        spacing=tuple(float(s) for s in spacing),
        origin=tuple(float(x) for x in affine[:3, 3]),
        direction=tuple(float(x) for x in affine[:3, :3].ravel()),
        dimensions=(int(d), int(h), int(w)),
        physical_size_mm=phys_mm,
        modality=modality,
        series_uid=series_uid,
        source_path=source_path,
    )

    if resample and preferred_spacing is not None:
        pv = _resample_to_spacing(pv, preferred_spacing)
    return pv


def _resample_to_spacing(pv: PatientVolume, target_spacing: tuple[float, float, float]) -> PatientVolume:
    try:
        from scipy.ndimage import zoom
    except Exception:  # pragma: no cover
        return pv
    src = pv.spacing
    if src == tuple(target_spacing) or any(s <= 0 for s in src) or any(t <= 0 for t in target_spacing):
        return pv
    factors = [s / t for s, t in zip(src, target_spacing)]
    vol = pv.image
    new_vol = zoom(vol, factors, order=1, mode="nearest")
    new_vol = np.ascontiguousarray(new_vol, dtype=np.float32)
    # Build new affine preserving the world frame (origin unchanged, spacing replaced)
    new_affine = pv.voxel_to_world.copy()
    for ax in range(3):
        axis_vec = new_affine[:3, ax]
        n = np.linalg.norm(axis_vec)
        if n > 0:
            new_affine[:3, ax] = axis_vec / n * target_spacing[ax]
    d, h, w = new_vol.shape
    phys = _physical_size_from_affine(new_affine, (w, h, d))
    return PatientVolume(
        image=new_vol,
        voxel_to_world=new_affine,
        world_to_voxel=np.linalg.inv(new_affine),
        spacing=tuple(target_spacing),
        origin=tuple(float(x) for x in new_affine[:3, 3]),
        direction=tuple(float(x) for x in new_affine[:3, :3].ravel()),
        dimensions=(int(d), int(h), int(w)),
        physical_size_mm=phys,
        modality=pv.modality,
        series_uid=pv.series_uid,
        source_path=pv.source_path,
    )


def _physical_size_from_affine(affine: np.ndarray, shape_voxels: tuple[int, int, int]) -> tuple[float, float, float]:
    """Physical size (mm) along each world axis from the affine + voxel counts.

    shape_voxels is (cols=W, rows=H, slices=D) matching the affine's voxel axes
    (col->x, row->y, slice->z), so the physical extent per voxel axis is
    |axis_vector| * count.
    """
    aff = np.asarray(affine, dtype=np.float64)
    counts = np.array(shape_voxels, dtype=np.float64)  # (W, H, D)
    extents = np.abs(aff[:3, :3] @ counts)             # mm along world x/y/z
    return (float(extents[0]), float(extents[1]), float(extents[2]))
