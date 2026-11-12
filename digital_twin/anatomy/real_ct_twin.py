"""Real CT-based digital twin rendering.

Two renderers:

1. ``render_real_ct_twin`` - renders an actual pancreatic-cancer CT (Medical
   Segmentation Decathlon Task 07) as orthogonal MPR (multi-planar reformat)
   slices plus the ground-truth pancreas and tumour meshes extracted from the
   exact same scan. Everything is in native millimetre coordinates, so what you
   see is a real patient's CT with the real tumour.

2. ``render_realistic_twin`` - the synthetic default: free real-organ meshes
   (NIH 3D Print Exchange / Visible Human project) arranged in the stylised twin
   space, with the pre-built tumourous pancreas from a real pancreatic-cancer
   CT. Used whenever no real CT file is available.

Sources (all free):
- MSD Task 07 Pancreas (CC-BY-SA-4.0): 420 CTs with pancreas + tumour labels.
- NIH 3D Print Exchange / Visible Human project organ STLs (CC BY).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

try:
    import nibabel as nib
except Exception:  # pragma: no cover - optional dependency
    nib = None

try:
    from skimage.measure import marching_cubes
except Exception:  # pragma: no cover - optional dependency
    marching_cubes = None

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover - optional dependency
    ndi = None

from .anatomy_loader import (
    load_organ_mesh, load_pancreas_tumor_mesh,
    load_prebuilt_tumored_pancreas,
)

MSD_DIR = os.path.join("DATASETS", "MSD_Pancreas")
DEFAULT_CASE = "409"


def _label_path(base: Path, image_name: str) -> Path:
    """Resolve both common MSD label naming conventions."""
    plain = base / "labelsTr" / image_name
    if plain.exists():
        return plain
    return base / "labelsTr" / image_name.replace(".nii.gz", "_label.nii.gz")


def discover_msd_cases(root: str | os.PathLike | None = None) -> list[str]:
    """Return MSD case ids that have both an image and a label NIfTI file."""
    base = Path(root or MSD_DIR)
    if not base.exists():
        return []
    cases = set()
    for im in base.glob("imagesTr/pancreas_*.nii.gz"):
        label = _label_path(base, im.name)
        if label.exists():
            cases.add(im.name[len("pancreas_"):-len(".nii.gz")])
    return sorted(cases, key=lambda x: (len(x), x))


def load_msd_case(case: str, root: str | os.PathLike | None = None
                  ) -> dict | None:
    """Load one MSD case as (volume HU, label, spacing mm, affine)."""
    if nib is None:
        return None
    base = Path(root or MSD_DIR)
    img_path = base / "imagesTr" / f"pancreas_{case}.nii.gz"
    lab_path = _label_path(base, img_path.name)
    if not img_path.exists() or not lab_path.exists():
        return None
    try:
        img = nib.load(str(img_path))
        lab = nib.load(str(lab_path))
        vol = img.get_fdata()
        label = lab.get_fdata()
        spacing = np.array(img.header.get_zooms()[:3], dtype=np.float64)
        return {"case": case, "volume": vol, "label": label,
                "spacing": spacing, "affine": img.affine,
                "path": str(img_path)}
    except Exception:
        return None


def _crop_roi(label: np.ndarray, margin: int = 20) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argwhere(label > 0)
    mn = np.clip(idx.min(axis=0) - margin, 0, None)
    mx = np.clip(idx.max(axis=0) + margin + 1, None, np.array(label.shape))
    s = tuple(slice(int(a), int(b)) for a, b in zip(mn, mx))
    return label[s], np.array(mn, dtype=np.int64)


def _crop_volume(volume: np.ndarray, offset: np.ndarray,
                 label_extent: np.ndarray, margin: int = 20) -> np.ndarray:
    """Crop the CT volume to exactly the annotation ROI (with margin), so its
    voxel grid starts at the origin of the mesh mm frame (offset * spacing)."""
    mx = np.clip(offset + label_extent + 1 + margin, None, np.array(volume.shape))
    s = tuple(slice(int(offset[i]), int(mx[i])) for i in range(3))
    return np.asarray(volume[s], dtype=np.float32)


def _mesh_from_mask(mask: np.ndarray, spacing: np.ndarray,
                    target_iso: float = 1.0, sigma: float = 0.8):
    """Marching-cubes mesh of a binary mask, returned in mm within a cropped ROI."""
    if marching_cubes is None or ndi is None:
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)
    scale = spacing / target_iso
    zoomed = ndi.zoom(mask.astype(np.float32), scale, order=0)
    smooth = ndi.gaussian_filter(zoomed.astype(np.float32), sigma=sigma)
    verts, faces, _, _ = marching_cubes(smooth, level=0.5)
    return verts * target_iso, faces.astype(np.int32)


def get_case_meshes(case: str, root: str | os.PathLike | None = None,
                    rebuild: bool = False) -> dict:
    """Cached ground-truth pancreas + tumour meshes (mm) for an MSD case.

    Also stores the cropped CT volume (downsampled, float16) in the same
    physical frame, so MPR slices and meshes align perfectly. The cache is
    written once per case; subsequent calls are fast.
    """
    base = Path(root or MSD_DIR)
    cache_path = base / f"meshes_{case}.npz"
    if cache_path.exists() and not rebuild:
        try:
            data = np.load(cache_path)
            if "vol_crop" in data.files:
                return {k: data[k] for k in data.files}
        except Exception:
            pass
    msd = load_msd_case(case, root)
    if msd is None:
        return {}
    label, offset = _crop_roi(msd["label"], margin=20)
    spacing = msd["spacing"]
    offset_mm = offset * spacing
    label_extent = np.array(label.shape, dtype=np.int64) - 1
    pan_v, pan_f = _mesh_from_mask((label == 1).astype(np.uint8), spacing)
    tum_v, tum_f = _mesh_from_mask((label == 2).astype(np.uint8), spacing)
    vol_crop = _crop_volume(msd["volume"], offset, label_extent, margin=20)
    out = {
        "vol_crop": vol_crop.astype(np.float16),
        "vol_shape": msd["volume"].shape,
        "spacing": spacing,
        "offset_mm": offset_mm,
        "pan_voxels": int((label == 1).sum()),
        "tum_voxels": int((label == 2).sum()),
        "pan_verts": pan_v + offset_mm, "pan_faces": pan_f,
        "tum_verts": tum_v + offset_mm, "tum_faces": tum_f,
    }
    try:
        base.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **out)
    except Exception:
        pass
    return out


def tumor_volume_cm3(case: str, root: str | os.PathLike | None = None) -> float:
    data = get_case_meshes(case, root)
    n_vox = int(data.get("tum_voxels", 0))
    if n_vox == 0:
        msd = load_msd_case(case, root)
        if msd is None:
            return 0.0
        n_vox = int((msd["label"] == 2).sum())
    spacing = data.get("spacing")
    if spacing is None:
        msd = load_msd_case(case, root)
        spacing = msd["spacing"] if msd else np.ones(3)
    return n_vox * float(np.prod(spacing)) / 1000.0


def _mpr_surface(vol: np.ndarray, spacing: np.ndarray,
                 plane: str, center: tuple[int, int, int],
                 extent: int = 110) -> go.Surface | None:
    """One orthogonal MPR slice through the CT volume as a colored Surface."""
    ny, nx, nz = vol.shape
    cy, cx, cz = center
    half = extent // 2
    rng_y = np.clip([cy - half, cy + half], 0, ny - 1)
    rng_x = np.clip([cx - half, cx + half], 0, nx - 1)
    rng_z = np.clip([cz - half, cz + half], 0, nz - 1)

    ys = np.arange(rng_y[0], rng_y[1] + 1)
    xs = np.arange(rng_x[0], rng_x[1] + 1)
    zs = np.arange(rng_z[0], rng_z[1] + 1)
    step = max(1, len(xs) // 220)

    sp_y, sp_x, sp_z = spacing
    if plane == "axial":
        z0 = int(cz)
        xx, yy = np.meshgrid(xs[::step], ys[::step])
        zz = np.full_like(xx, z0)
        vals = vol[ys[::step][:, None], xs[::step][None, :], z0]
        xyz = np.stack([xx * sp_x, yy * sp_y, zz * sp_z], axis=-1)
    elif plane == "coronal":
        y0 = int(cy)
        xx, zz = np.meshgrid(xs[::step], zs[::step])
        yy = np.full_like(xx, y0)
        vals = vol[y0, xs[::step][None, :], zs[::step][:, None]]
        xyz = np.stack([xx * sp_x, yy * sp_y, zz * sp_z], axis=-1)
    else:  # sagittal
        x0 = int(cx)
        yy, zz = np.meshgrid(ys[::step], zs[::step])
        xx = np.full_like(yy, x0)
        vals = vol[ys[::step][None, :], x0, zs[::step][:, None]]
        xyz = np.stack([xx * sp_x, yy * sp_y, zz * sp_z], axis=-1)

    vmin, vmax = float(np.percentile(vals, 2)), float(np.percentile(vals, 99.5))
    if vmax - vmin < 1e-6:
        vmax = vmin + 1
    vals_c = np.clip(vals, vmin, vmax)
    return go.Surface(
        x=xyz[..., 0], y=xyz[..., 1], z=xyz[..., 2],
        surfacecolor=vals_c,
        colorscale="gray",
        cmin=vmin, cmax=vmax,
        opacity=0.62,
        showscale=False,
        contours=dict(z=dict(show=False), y=dict(show=False), x=dict(show=False)),
        hoverinfo="skip",
        lighting=dict(ambient=0.9, diffuse=0.4, specular=0.0, roughness=1.0),
    )


def render_real_ct_twin(case: str | None = None,
                        root: str | os.PathLike | None = None) -> tuple[go.Figure, dict]:
    """Build the Real CT Twin: real patient CT MPR slices + ground-truth
    pancreas & tumour meshes, in native mm coordinates."""
    if case is None:
        cases = discover_msd_cases(root)
        if not cases:
            return go.Figure(), {"case": None, "error": "no MSD cases found"}
        case = cases[-1] if DEFAULT_CASE not in cases else DEFAULT_CASE
    msd = load_msd_case(case, root)
    if msd is None:
        return go.Figure(), {"case": case, "error": f"case {case} not found"}
    data = get_case_meshes(case, root)
    vol = np.asarray(data["vol_crop"], dtype=np.float32)
    spacing = np.asarray(data["spacing"])
    offset_mm = np.asarray(data["offset_mm"])
    # work in crop-local mm (MPR frame); meshes are stored in full-volume mm
    pan = data.get("pan_verts")
    tum = data.get("tum_verts")
    pan_local = pan - offset_mm if pan is not None and len(pan) else pan
    tum_local = tum - offset_mm if tum is not None and len(tum) else tum
    if tum_local is not None and len(tum_local):
        center = tuple(int(v) for v in np.clip(tum_local.mean(0) / spacing, 0, None))
    else:
        center = tuple(int(s // 2) for s in vol.shape)

    traces: list = []
    for plane in ("axial", "coronal", "sagittal"):
        s = _mpr_surface(vol, spacing, plane, center)
        if s is not None:
            traces.append(s)

    if pan_local is not None and len(pan_local):
        traces.append(go.Mesh3d(
            x=pan_local[:, 0], y=pan_local[:, 1], z=pan_local[:, 2],
            i=data["pan_faces"][:, 0], j=data["pan_faces"][:, 1],
            k=data["pan_faces"][:, 2],
            opacity=0.88, color="#e0a468", showscale=False,
            lighting=dict(ambient=0.45, diffuse=0.85, specular=0.25,
                          roughness=0.4, fresnel=0.1),
            name="Pancreas (ground truth)",
            hoverinfo="name", text=f"<b>Pancreas</b> — case {case}<br>real CT segmentation",
        ))
    if tum_local is not None and len(tum_local):
        traces.append(go.Mesh3d(
            x=tum_local[:, 0], y=tum_local[:, 1], z=tum_local[:, 2],
            i=data["tum_faces"][:, 0], j=data["tum_faces"][:, 1],
            k=data["tum_faces"][:, 2],
            opacity=1.0, color="#ff3b3b", showscale=False,
            lighting=dict(ambient=0.5, diffuse=0.8, specular=0.4,
                          roughness=0.3, fresnel=0.2),
            name="Tumour (ground truth)",
            hoverinfo="text",
            text=f"<b>Tumour</b> — case {case}<br>volume {tumor_volume_cm3(case, root):.1f} cm³",
        ))

    vol_cm3 = tumor_volume_cm3(case, root)
    fig = go.Figure(data=traces)
    vol_span = np.array(vol.shape) * spacing
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showgrid=False, range=[0, vol_span[1]]),
            yaxis=dict(visible=False, showgrid=False, range=[0, vol_span[0]]),
            zaxis=dict(visible=False, showgrid=False, range=[0, vol_span[2]]),
            bgcolor="rgba(5,8,12,0.35)",
            camera=dict(eye=dict(x=1.6, y=-1.5, z=0.7),
                        up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0)),
            aspectmode="data",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0), height=760, showlegend=True,
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0.45)",
                    font=dict(color="#dbe4ee", size=11)),
    )
    meta = {"case": case,
            "pan_voxels": int(data.get("pan_voxels", 0)),
            "tum_voxels": int(data.get("tum_voxels", 0)),
            "tumor_volume_cm3": vol_cm3,
            "spacing": tuple(spacing.tolist()),
            "slices": int(data["vol_shape"][2]),
            "source": "MSD Task07 Pancreas (real CT)"}
    return fig, meta


def _tumor_trace(organs) -> list:
    traces = []
    panc = organs.get("Pancreas")
    if panc is not None and panc.risk_score > 25:
        tum = load_pancreas_tumor_mesh()
        if len(tum["vertices"]) > 0:
            opacity = 0.75 + min(0.25, panc.risk_score / 400.0)
            traces.append(go.Mesh3d(
                x=tum["vertices"][:, 0], y=tum["vertices"][:, 1],
                z=tum["vertices"][:, 2],
                i=tum["faces"][:, 0], j=tum["faces"][:, 1], k=tum["faces"][:, 2],
                opacity=opacity, color="#ff3b3b", showscale=False,
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.4,
                              roughness=0.3, fresnel=0.2),
                name="Pancreatic tumour",
                hoverinfo="text",
                text=("<b>Pancreatic tumour</b><br>pre-built model from real "
                      "pancreatic-cancer CT (MSD case 409)"),
            ))
    return traces


def render_realistic_twin(organs, outline: bool = True) -> list:
    """Synthetic default: real free organ meshes + pre-built tumourous pancreas
    from a real pancreatic-cancer CT, in the stylised twin coordinate space."""
    from .organ_renderer import _torso_outline, _organ_to_mesh3d, REALISTIC_ORGAN_COLORS
    traces = [_torso_outline()] if outline else []
    for name, organ in organs.items():
        base = REALISTIC_ORGAN_COLORS.get(name, organ.colour)
        t = _organ_to_mesh3d(organ, realistic=True)
        t.text = f"<b>{name}</b><br><i>Real anatomical mesh</i>"
        traces.append(t)
    traces.extend(_tumor_trace(organs))
    return traces
