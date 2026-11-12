"""Export Digital-Twin 3D meshes (Real CT Twin + Realistic Anatomy) as JSON
for the React/R3F frontend.

Uses the shared renderers in ``digital_twin.anatomy`` so the API and frontend show
the same patient-specific models:
- Real CT Twin: ground-truth pancreas + tumour meshes from a real MSD Task07
  Pancreatic-cancer CT, in crop-local millimetre coordinates.
- Realistic Anatomy: free real-organ meshes (NIH 3D Print Exchange / Visible
  Human project) plus the pre-built tumourous pancreas, in the stylised twin
  coordinate space.

Each organ is emitted as ``{positions, indices}`` (flat arrays, mm) — the same
schema ``MedicalImageViewer.tsx`` already consumes. Meshes are decimated to
web-friendly densities and the results are cached in memory.
"""
from __future__ import annotations

import logging
import threading

import numpy as np

logger = logging.getLogger(__name__)

REALISTIC_ORGAN_ORDER = [
    "Pancreas", "Liver", "Gallbladder", "Bile Duct",
    "Duodenum", "Stomach", "Spleen", "Lungs",
    "Lymph Nodes", "Heart", "Kidneys",
]

REAL_CT_PANCREAS_COLOR = "#e0a468"
REAL_CT_TUMOR_COLOR = "#ff3b3b"

# Web-friendly density targets (vert counts).
_REALISTIC_TARGET = 12000
_REALCT_TARGET = 15000

_realistic_cache: dict | None = None
_realct_cache: dict[str, dict] = {}
_lock = threading.Lock()


def _decimate_for_web(verts, faces, target: int):
    """Reduce a mesh to at most `target` vertices for the browser."""
    from digital_twin.anatomy.mesh_utils import decimate_mesh
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    if len(v) > target:
        return decimate_mesh(v, f, target_verts=target)
    return v, f


def _mesh_payload(vertices, faces, color: str, name: str,
                  opacity: float = 0.95, colors=None, kind: str = "solid") -> dict:
    """One organ/tumour mesh in the {positions, indices} web format.

    ``colors`` (optional) is an (N, 3) array of per-vertex RGB floats in
    [0, 1] — matches the vertex-color range used by the shared renderer
    (e.g. ``render_normal``) so the browser shows the same realistic tissue
    shading. ``kind`` is ``"solid"`` (indexed triangle mesh) or ``"lines"``
    (consecutive position pairs, no index).
    """
    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError(f"mesh {name!r}: bad vertex array {verts.shape}")
    payload = {
        "name": name,
        "color": color,
        "opacity": float(opacity),
        "kind": kind,
        # round to 0.001 mm so the JSON payload stays small on the wire
        "positions": np.round(verts, 3).reshape(-1).tolist(),
    }
    if kind != "lines":
        tris = np.asarray(faces, dtype=np.int32)
        if tris.ndim != 2 or tris.shape[1] != 3:
            raise ValueError(f"mesh {name!r}: bad face array {tris.shape}")
        payload["indices"] = tris.reshape(-1).tolist()
    if colors is not None:
        cols = np.asarray(colors, dtype=np.float64)
        if cols.ndim != 2 or cols.shape[1] != 3 or len(cols) != len(verts):
            raise ValueError(f"mesh {name!r}: bad vertex color array {cols.shape}")
        payload["colors"] = np.round(np.clip(cols, 0.0, 1.0), 4).reshape(-1).tolist()
    return payload


def available_msd_cases() -> list[str]:
    from digital_twin.anatomy.real_ct_twin import discover_msd_cases
    return discover_msd_cases()


_TORSO_COLOR = "#60a5fa"


def _torso_outline_meshes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The translucent body outline used by the shared 3D renderer
    (``organ_renderer._torso_outline``) as a triangle surface + contour lines.

    Returns ``(verts, tris, line_points)`` in twin coordinate space; the solid
    surface is emitted at low opacity and the contours as a ``"lines"`` mesh.
    """
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(-1, 1, 60)
    U, V = np.meshgrid(u, v)
    w = 1.5 * (1 - 0.12 * V ** 2 + 0.04 * V ** 4)
    d = 0.75 * (1 - 0.10 * V ** 2 + 0.02 * V ** 4)
    X = w * np.cos(U)
    Y = d * np.sin(U)
    Z = 2.2 * V
    shoulder = Z > 1.6
    X[shoulder] *= 1 + 0.15 * (Z[shoulder] - 1.6) / 0.6

    rows, cols = Z.shape
    verts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    grid = np.arange(rows * cols).reshape(rows, cols)
    tris: list[list[int]] = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            a, b, c, dd = grid[i, j], grid[i, j + 1], grid[i + 1, j], grid[i + 1, j + 1]
            tris.append([a, c, b])
            tris.append([b, c, dd])
    lines: list[list[int]] = []
    for i in range(rows):
        for j in range(cols - 1):
            lines.append([grid[i, j], grid[i, j + 1]])
    for j in range(cols):
        for i in range(rows - 1):
            lines.append([grid[i, j], grid[i + 1, j]])
    line_points = verts[np.asarray(lines, dtype=np.int64)].reshape(-1, 3)
    return verts, np.array(tris, dtype=np.int32), line_points


def realistic_anatomy_payload() -> dict:
    """All default twin organs with real meshes, in twin coordinate space."""
    global _realistic_cache
    if _realistic_cache is not None:
        return _realistic_cache

    from digital_twin.anatomy.anatomy_loader import (
        load_organ_mesh, load_pancreas_tumor_mesh,
    )
    from digital_twin.anatomy.organ_renderer import (
        REALISTIC_ORGAN_COLORS, _per_vertex_color,
    )

    organs: list[dict] = []
    with _lock:
        if _realistic_cache is not None:
            return _realistic_cache

        # Translucent body outline matching the shared `_torso_outline`.
        torso_verts, torso_tris, torso_lines = _torso_outline_meshes()
        organs.append(_mesh_payload(torso_verts, torso_tris, _TORSO_COLOR,
                                    "Torso outline", opacity=0.045))
        organs.append({
            "name": "Torso contours",
            "color": _TORSO_COLOR,
            "opacity": 0.12,
            "kind": "lines",
            "legend": False,
            "positions": np.round(torso_lines, 3).reshape(-1).tolist(),
        })

        for name in REALISTIC_ORGAN_ORDER:
            try:
                mesh = load_organ_mesh(name)
            except Exception:
                logger.warning("load_organ_mesh(%r) failed; skipping",
                               name, exc_info=True)
                continue
            verts = mesh.get("vertices")
            faces = mesh.get("faces")
            if verts is None or faces is None or len(verts) == 0:
                continue
            verts, faces = _decimate_for_web(verts, faces, _REALISTIC_TARGET)
            base = REALISTIC_ORGAN_COLORS.get(name, "#5db7a2")
            # Per-vertex realistic tissue shading, same as `render_normal`.
            colors = _per_vertex_color(base, verts, noise_amp=10,
                                       vary_lighting=True)
            organs.append(_mesh_payload(
                verts, faces, base, name, opacity=0.95, colors=colors,
            ))

        try:
            tum = load_pancreas_tumor_mesh()
            if len(tum["vertices"]) > 0:
                v, f = _decimate_for_web(tum["vertices"], tum["faces"],
                                         _REALISTIC_TARGET)
                organs.append(_mesh_payload(v, f, "#ff3b3b",
                                            "Pancreatic tumour", opacity=0.95))
        except Exception:
            logger.warning("load_pancreas_tumor_mesh failed", exc_info=True)

        _realistic_cache = {
            "mode": "realistic",
            "title": "Realistic Anatomy",
            "organs": organs,
            "meta": {
                "source": "NIH 3D Print Exchange / Visible Human project "
                          "(CC BY) + pre-built tumourous pancreas from MSD case 409",
                "note": "Real free anatomical meshes placed in the stylised "
                        "twin space, shaded like the shared 3D "
                        "'Normal Anatomy' renderer.",
            },
        }
    return _realistic_cache


def real_ct_payload(case: str) -> dict:
    """Ground-truth pancreas + tumour meshes for one MSD Task07 case, in the
    crop-local millimetre frame used by the frontend MPR rendering.

    Serves the compiled ``meshes_<case>.npz`` cache (pre-extracted from a real
    MSD CT) when present, so the twin renders from the patient's actual CT even
    when the raw NIfTI image/label files are not installed alongside. Only
    rebuilds from the NIfTI when the cache is missing.
    """
    from digital_twin.anatomy.real_ct_twin import get_case_meshes, tumor_volume_cm3

    cached = _realct_cache.get(case)
    if cached is not None:
        return cached

    data = get_case_meshes(case)
    if not data:
        cases = available_msd_cases()
        raise ValueError(
            f"MSD case {case!r} not found; available: {', '.join(cases) or 'none'}"
        )

    with _lock:
        cached = _realct_cache.get(case)
        if cached is not None:
            return cached
        offset_mm = np.asarray(data["offset_mm"], dtype=np.float64)

        organs: list[dict] = []
        pan = np.asarray(data["pan_verts"], dtype=np.float64) - offset_mm
        if len(pan) and len(data["pan_faces"]):
            pv, pf = _decimate_for_web(pan, data["pan_faces"], _REALCT_TARGET)
            organs.append(_mesh_payload(pv, pf, REAL_CT_PANCREAS_COLOR,
                                        "Pancreas (ground truth)", opacity=0.88))
        tum = np.asarray(data["tum_verts"], dtype=np.float64) - offset_mm
        tumor_centroid: list[float] = [0.0, 0.0, 0.0]
        tumor_bounds_mm: list[list[float]] = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        if len(tum):
            tumor_centroid = [round(float(v), 4) for v in tum.mean(axis=0)]
            tumor_bounds_mm = [
                [round(float(v), 4) for v in tum.min(axis=0)],
                [round(float(v), 4) for v in tum.max(axis=0)],
            ]
            organs.append(_mesh_payload(
                *_decimate_for_web(tum, data["tum_faces"], _REALCT_TARGET),
                REAL_CT_TUMOR_COLOR, "Tumour (ground truth)",
            ))

        spacing = np.asarray(data["spacing"], dtype=np.float64)
        vol_shape = np.asarray(data["vol_shape"], dtype=np.int64)
        payload = {
            "mode": "real-ct",
            "title": f"Real CT Twin — MSD case {case}",
            "organs": organs,
            "meta": {
                "case": case,
                "source": "MSD Task07 Pancreas (real CT)",
                "pan_voxels": int(data.get("pan_voxels", 0)),
                "tum_voxels": int(data.get("tum_voxels", 0)),
                "tumor_volume_cm3": round(tumor_volume_cm3(case), 2),
                "tumor_centroid": tumor_centroid,
                "tumor_bounds_mm": tumor_bounds_mm,
                "slices": int(vol_shape[2]) if vol_shape.size >= 3 else 0,
                "spacing_mm": [round(float(x), 4) for x in spacing],
            },
        }
        _realct_cache[case] = payload
    return payload
