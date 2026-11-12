import os
import numpy as np
from .mesh_utils import (
    ellipsoid_mesh, tube_mesh, merge_meshes,
    decimate_mesh, laplacian_smooth, place_mesh, mirror_mesh,
    keep_largest_component,
)
from .mesh_io import load_mesh_file

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MESH_DIR = os.path.join(os.path.dirname(HERE), "models", "meshes")
PROCESSED_DIR = os.path.join(MESH_DIR, "_processed")

_HAS_PYVISTA = False
_HAS_TRIMESH = False
try:
    import pyvista as pv
    _HAS_PYVISTA = True
except ImportError:
    pass
try:
    import trimesh
    _HAS_TRIMESH = True
except ImportError:
    pass

_MESH_CACHE: dict[str, dict] = {}
SEED = 42

# --- Real anatomical mesh assets (NIH 3D Print Exchange / Visible Human project,
# free, CC BY). Each entry: (asset filename, canonical centre, scale, needs-mirror)
REAL_MESH_ASSETS: dict[str, tuple[str, tuple, float, bool]] = {
    "liver":     ("liver.stl",           (0.55, 0.18, 0.55),  1.00, False),
    "spleen":    ("spleen.stl",          (-1.02, -0.08, 0.35), 0.42, False),
    "stomach":   ("stomach.stl",         (-0.26, 0.26, 0.12), 0.75, False),
    "heart":     ("heart.stl",           (-0.10, 0.22, 1.00), 0.52, False),
    "kidney":    ("kidney_left.stl",     (-0.55, -0.18, 0.0), 0.26, False),
    "lung":      ("lung_left.stl",       (-0.62, 0.04, 1.15), 0.95, False),
    "pancreas":  ("pancreas_healthy.stl", (0.0, -0.10, 0.05), 0.95, False),
}

# Organs rendered from two mirrored real halves (left = source asset).
DUAL_ORGAN_ASSETS: dict[str, tuple[str, tuple, float, tuple]] = {
    "kidneys": ("kidney_left.stl", (0.0, -0.18, 0.0), 0.26, (-0.55, 0.55)),
    "lungs":   ("lung_left.stl",   (0.0, 0.04, 1.15), 0.95, (-0.62, 0.62)),
}


def load_organ_mesh(organ_name: str) -> dict:
    key = organ_name.lower().strip()
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]
    if key == "pancreas":
        # default pancreas = pre-built tumorous model from a real
        # pancreatic-cancer CT (the pancreas part, placed canonically)
        pre = load_prebuilt_tumored_pancreas()
        result = {"vertices": pre["pan_verts"], "faces": pre["pan_faces"],
                  "source": "prebuilt_ct", "path": "pancreas_tumored.npz"}
        _MESH_CACHE[key] = result
        return result
    if key in DUAL_ORGAN_ASSETS:
        result = _load_dual_real_asset(key)
        if result:
            _MESH_CACHE[key] = result
            return result
    if key in REAL_MESH_ASSETS:
        result = _load_real_mesh_asset(key)
        if result:
            _MESH_CACHE[key] = result
            return result
    for ext in (".stl", ".obj", ".glb", ".gltf"):
        fname = key.replace(" ", "_") + ext
        path = os.path.join(MESH_DIR, fname)
        if os.path.isfile(path):
            result = _load_file(path)
            if result:
                _MESH_CACHE[key] = result
                return result
    if _HAS_PYVISTA:
        result = _load_pyvista(key)
        if result:
            _MESH_CACHE[key] = result
            return result
    if _HAS_TRIMESH:
        result = _load_trimesh(key)
        if result:
            _MESH_CACHE[key] = result
            return result
    result = _procedural_organ_mesh(key)
    _MESH_CACHE[key] = result
    return result


def load_pancreas_tumor_mesh() -> dict:
    """Tumor mesh of the pre-built tumorous pancreas (placed canonically)."""
    cache_key = "__prebuilt_tumor_mesh"
    if cache_key in _MESH_CACHE:
        return _MESH_CACHE[cache_key]
    pre = load_prebuilt_tumored_pancreas()
    result = {"vertices": pre["tum_verts"], "faces": pre["tum_faces"],
              "source": "prebuilt_ct", "path": "pancreas_tumored.npz"}
    _MESH_CACHE[cache_key] = result
    return result


def _load_real_mesh_asset(key: str) -> dict | None:
    """Load a free real-organ mesh, decimate + smooth it, and place it at its
    canonical position in the twin coordinate space (results cached on disk)."""
    fname, centre, scale, mirror = REAL_MESH_ASSETS[key]
    cached = _load_processed_cache(key)
    if cached is not None:
        return cached
    raw = load_mesh_file(os.path.join(MESH_DIR, fname))
    if raw is None:
        return None
    verts = np.asarray(raw["vertices"], dtype=np.float64)
    faces = np.asarray(raw["faces"], dtype=np.int32)
    if len(verts) > 60000:
        verts, faces = decimate_mesh(verts, faces, target_verts=40000)
    verts, faces = keep_largest_component(verts, faces)
    verts = laplacian_smooth(verts, faces, iterations=3, factor=0.5)
    verts = place_mesh(verts, centre, scale=scale)
    if mirror:
        verts = mirror_mesh(verts, axis=0)
    result = {"vertices": verts, "faces": faces,
              "source": "real_stl", "path": fname}
    _save_processed_cache(key, result)
    return result


def _load_dual_real_asset(key: str) -> dict | None:
    """Build a paired left/right organ from a single real half-organ mesh."""
    fname, centre, scale, xs = DUAL_ORGAN_ASSETS[key]
    cached = _load_processed_cache(key)
    if cached is not None:
        return cached
    raw = load_mesh_file(os.path.join(MESH_DIR, fname))
    if raw is None:
        return None
    verts = np.asarray(raw["vertices"], dtype=np.float64)
    faces = np.asarray(raw["faces"], dtype=np.int32)
    if len(verts) > 60000:
        verts, faces = decimate_mesh(verts, faces, target_verts=40000)
    verts, faces = keep_largest_component(verts, faces)
    verts = laplacian_smooth(verts, faces, iterations=3, factor=0.5)
    half = place_mesh(verts, (xs[0], centre[1], centre[2]), scale=scale)
    mirror_half = mirror_mesh(half, axis=0)
    mirror_half[:, 0] += (xs[1] - xs[0])
    all_v = np.vstack([half, mirror_half])
    all_f = np.vstack([faces, faces + len(half)])
    result = {"vertices": all_v, "faces": all_f,
              "source": "real_stl_dual", "path": fname}
    _save_processed_cache(key, result)
    return result


def _processed_path(key: str) -> str:
    safe = key.replace(" ", "_")
    return os.path.join(PROCESSED_DIR, f"{safe}.npz")


def _load_processed_cache(key: str) -> dict | None:
    path = _processed_path(key)
    if not os.path.isfile(path):
        return None
    try:
        data = np.load(path, allow_pickle=False)
        return {"vertices": data["verts"], "faces": data["faces"],
                "source": "real_stl", "path": key}
    except Exception:
        return None


def _save_processed_cache(key: str, mesh: dict) -> None:
    try:
        os.makedirs(PROCESSED_DIR, exist_ok=True)
        np.savez_compressed(_processed_path(key),
                            verts=mesh["vertices"], faces=mesh["faces"])
    except Exception:
        pass


def load_prebuilt_tumored_pancreas() -> dict:
    """Pre-built tumorous pancreas 3D model (from a real pancreatic-cancer CT,
    MSD Task 07 case 409), normalized into the twin coordinate space."""
    cache_key = "__prebuilt_tumored_pancreas"
    if cache_key in _MESH_CACHE:
        return _MESH_CACHE[cache_key]
    path = os.path.join(MESH_DIR, "pancreas_tumored.npz")
    result = {
        "pan_verts": np.empty((0, 3)), "pan_faces": np.empty((0, 3), dtype=np.int32),
        "tum_verts": np.empty((0, 3)), "tum_faces": np.empty((0, 3), dtype=np.int32),
    }
    if os.path.isfile(path):
        try:
            data = np.load(path, allow_pickle=False)
            cx, cy, cz = REAL_MESH_ASSETS["pancreas"][1]
            scale = REAL_MESH_ASSETS["pancreas"][2]
            pan = data["pan_verts"] * scale + np.array([cx, cy, cz])
            tum = data["tum_verts"] * scale + np.array([cx, cy, cz])
            result = {
                "pan_verts": pan, "pan_faces": data["pan_faces"],
                "tum_verts": tum, "tum_faces": data["tum_faces"],
            }
        except Exception:
            pass
    _MESH_CACHE[cache_key] = result
    return result


def _load_file(path: str) -> dict | None:
    try:
        if _HAS_PYVISTA:
            import pyvista as pv
            m = pv.read(path)
            return dict(vertices=np.array(m.points, dtype=np.float64),
                        faces=m.faces.reshape(-1, 4)[:, 1:],
                        source="file", path=path)
    except Exception:
        pass
    try:
        import trimesh
        m = trimesh.load(path)
        return dict(vertices=np.array(m.vertices, dtype=np.float64),
                    faces=np.array(m.faces, dtype=np.int32),
                    source="file", path=path)
    except Exception:
        pass
    return None


def _load_pyvista(key: str) -> dict | None:
    known = {"pancreas": "pancreas.stl", "liver": "liver.stl"}
    fname = known.get(key)
    if not fname:
        return None
    for base in (MESH_DIR, "/usr/share/medical-meshes", "./meshes"):
        p = os.path.join(base, fname)
        if os.path.isfile(p):
            return _load_file(p)
    return None


def _load_trimesh(key: str) -> dict | None:
    known = {"pancreas": "pancreas.stl", "liver": "liver.stl"}
    fname = known.get(key)
    if not fname:
        return None
    for base in (MESH_DIR, "/usr/share/medical-meshes", "./meshes"):
        p = os.path.join(base, fname)
        if os.path.isfile(p):
            return _load_file(p)
    return None


def _pancreas():
    n = 80
    s_ = np.linspace(0, 1, n)
    cx = 0.65 - 1.6 * s_ + 0.35 * s_ ** 2 + 0.08 * np.sin(s_ * 6)
    cy = 0.05 + 0.45 * s_ - 0.65 * s_ ** 2 + 0.04 * np.cos(s_ * 5)
    cz = -0.10 + 0.50 * s_ + 0.03 * np.sin(s_ * 4)
    radii = np.zeros(n)
    for i, si in enumerate(s_):
        if si < 0.15:
            t = si / 0.15
            radii[i] = 0.22 * (1 + 0.25 * np.sin(t * np.pi))
        elif si < 0.22:
            radii[i] = 0.22 - (si - 0.15) / 0.07 * 0.10
        elif si < 0.30:
            radii[i] = 0.12 + 0.02 * np.sin(si * 20)
        elif si < 0.70:
            radii[i] = 0.09 + 0.015 * np.sin(si * 15)
        else:
            radii[i] = 0.09 * (1 - (si - 0.70) / 0.30 * 0.55) + 0.01 * np.sin(si * 10)
    return tube_mesh(list(zip(cx, cy, cz)), radii, n_circ=28, noise_amp=0.012, noise_freq=6.0, seed=SEED)


def _liver():
    def deform(x, y, z):
        dx = 0.12 * z + 0.04 * np.sin(z * 4)
        dy = 0.04 * y + 0.03 * np.cos(x * 3)
        dz = -0.03 * x ** 2 - 0.02 * np.sin(y * 5)
        if x > 0:
            dx += -0.18 - 0.03 * np.sin(z * 3 + 1)
            dy += 0.06
            dz += -0.06 * x
        return (dx, dy, dz)
    return ellipsoid_mesh(0.55, 0.18, 0.55, 0.52, 0.22, 0.32, deform,
                          n_u=48, n_v=32, noise_amp=0.015, noise_freq=5.0, seed=SEED)


def _gallbladder():
    verts, _ = ellipsoid_mesh(0.78, 0.32, 0.28, 0.08, 0.08, 0.12,
                              n_u=28, n_v=20, noise_amp=0.005, noise_freq=8.0, seed=SEED)
    for i in range(len(verts)):
        x, y, z = verts[i]
        dy = y - 0.32
        factor = 1 + 0.35 * (z - 0.28) / 0.12
        verts[i] = [x, 0.32 + dy * factor, z]
    return verts, _mesh_grid_faces(28, 20)


def _bile_duct():
    pa = [[0.48, 0.28, 0.32], [0.50, 0.26, 0.22],
          [0.52, 0.24, 0.12], [0.52, 0.22, 0.04],
          [0.50, 0.20, -0.02]]
    return tube_mesh(pa, np.linspace(0.035, 0.025, 5), n_circ=14, noise_amp=0.003)


def _duodenum():
    pa = [[0.42, 0.14, 0.22], [0.52, 0.18, 0.14], [0.56, 0.20, 0.04],
          [0.58, 0.18, -0.04], [0.56, 0.14, -0.12], [0.52, 0.08, -0.18],
          [0.46, 0.04, -0.24], [0.38, 0.0, -0.26], [0.30, -0.04, -0.24],
          [0.24, -0.06, -0.20], [0.20, -0.06, -0.14]]
    return tube_mesh(pa, np.linspace(0.08, 0.055, 11), n_circ=18, noise_amp=0.008, noise_freq=7.0, seed=SEED)


def _stomach():
    pa = [[-0.18, 0.42, 0.50], [-0.22, 0.46, 0.40], [-0.28, 0.46, 0.30],
          [-0.32, 0.42, 0.20], [-0.38, 0.34, 0.10], [-0.34, 0.20, 0.0],
          [-0.24, 0.10, -0.08], [-0.14, 0.02, -0.14], [-0.06, -0.02, -0.18],
          [0.04, -0.02, -0.18], [0.12, 0.02, -0.14]]
    radii = np.linspace(0.24, 0.14, 11)
    radii[3:7] *= 1.15
    return tube_mesh(pa, radii, n_circ=22, noise_amp=0.015, noise_freq=5.0, seed=SEED)


def _spleen():
    def deform(x, y, z):
        return (0.18 * z + 0.03 * np.sin(z * 5),
                -0.12 * x + 0.04 * np.sin(z * 5) + 0.02 * np.cos(x * 4),
                0.02 * np.sin(x * 3))
    return ellipsoid_mesh(-1.02, -0.08, 0.35, 0.18, 0.10, 0.26, deform,
                          n_u=34, n_v=22, noise_amp=0.01, noise_freq=6.0, seed=SEED)


def _lungs():
    left_verts, left_tris = ellipsoid_mesh(
        -0.62, 0.04, 1.15, 0.38, 0.22, 0.55,
        n_u=36, n_v=24, noise_amp=0.012, noise_freq=5.0, seed=SEED)
    right_verts, right_tris = ellipsoid_mesh(
        0.62, 0.04, 1.15, 0.38, 0.22, 0.55,
        n_u=36, n_v=24, noise_amp=0.012, noise_freq=5.0, seed=SEED + 1)
    for i in range(len(left_verts)):
        x, y, z = left_verts[i]
        if y > 0 and z < 0.15:
            left_verts[i, 1] += 0.08
    right_tris += len(left_verts)
    verts = np.vstack([left_verts, right_verts])
    tris = np.vstack([left_tris, right_tris])
    return verts, tris


def _heart():
    def deform(x, y, z):
        return (0.06 * x ** 2 + 0.02 * np.sin(z * 3),
                0.04 * z + 0.015 * np.cos(x * 4),
                -0.04 * y ** 2 - 0.01 * np.sin(x * 3))
    return ellipsoid_mesh(-0.10, 0.22, 1.0, 0.24, 0.18, 0.30, deform,
                          n_u=32, n_v=22, noise_amp=0.008, noise_freq=7.0, seed=SEED)


def _kidneys():
    def deform(x, y, z):
        return (0.01 * np.sin(z * 4), 0.01 * np.cos(x * 3), 0)
    left_v, left_t = ellipsoid_mesh(
        -0.55, -0.18, 0.0, 0.12, 0.07, 0.18, deform,
        n_u=28, n_v=20, noise_amp=0.006, noise_freq=7.0, seed=SEED)
    right_v, right_t = ellipsoid_mesh(
        0.55, -0.18, 0.0, 0.12, 0.07, 0.18, deform,
        n_u=28, n_v=20, noise_amp=0.006, noise_freq=7.0, seed=SEED + 1)
    right_t += len(left_v)
    verts = np.vstack([left_v, right_v])
    tris = np.vstack([left_t, right_t])
    return verts, tris


def _lymph_nodes():
    sites = [[0.18, -0.12, 0.08], [-0.08, -0.08, 0.04],
             [-0.28, -0.18, 0.12], [0.48, -0.08, -0.12],
             [-0.48, -0.12, 0.18], [0.0, -0.28, -0.08],
             [0.65, 0.02, 0.55], [-0.58, 0.02, 0.75],
             [0.28, 0.08, 0.85], [-0.28, 0.08, 0.85]]
    meshes = []
    for sx, sy, sz in sites:
        meshes.append(ellipsoid_mesh(sx, sy, sz, 0.035, 0.035, 0.035,
                                     n_u=16, n_v=12, noise_amp=0.004, noise_freq=8.0, seed=SEED))
    return merge_meshes(meshes)


def _blood_vessels():
    import warnings
    pts = []
    for z in np.linspace(-1.2, 1.8, 30):
        r = 0.05 + 0.015 * (z + 1.2) / 3.0
        for _ in range(8):
            a = np.random.uniform(0, 2 * np.pi)
            pts.append([0.0 + r * np.cos(a), -0.32 + r * np.sin(a), z])
    for t in np.linspace(0, 1, 18):
        x = 0.45 * t
        y = -0.08 + 0.08 * t
        z = 0.45 - 0.25 * t
        r = 0.035 * (1 - 0.3 * t)
        for _ in range(6):
            a = np.random.uniform(0, 2 * np.pi)
            pts.append([x + r * np.cos(a), y + r * np.sin(a), z])
    verts = np.array(pts, dtype=np.float64)
    return verts, None


def _mesh_grid_faces(n_u, n_v):
    tris = []
    for a in range(n_u - 1):
        for b in range(n_v - 1):
            p1 = a * n_v + b
            p2 = (a + 1) * n_v + b
            p3 = (a + 1) * n_v + (b + 1)
            p4 = a * n_v + (b + 1)
            tris.append([p1, p2, p3])
            tris.append([p1, p3, p4])
    return np.array(tris, dtype=np.int32)


def _procedural_organ_mesh(key: str) -> dict:
    gen = _GENERATORS.get(key)
    if gen is None:
        verts, _ = ellipsoid_mesh(0, 0, 0, 0.2, 0.2, 0.2, n_u=24, n_v=16)
        return dict(vertices=verts, faces=_mesh_grid_faces(24, 16),
                    source="fallback", path=None)
    verts, faces = gen()
    return dict(vertices=np.array(verts, dtype=np.float64),
                faces=np.array(faces, dtype=np.int32) if faces is not None else None,
                source="procedural", path=None)


_GENERATORS = {
    "pancreas": _pancreas,
    "liver": _liver,
    "gallbladder": _gallbladder,
    "bile_duct": _bile_duct,
    "bile duct": _bile_duct,
    "duodenum": _duodenum,
    "stomach": _stomach,
    "spleen": _spleen,
    "lungs": _lungs,
    "lung": _lungs,
    "heart": _heart,
    "kidneys": _kidneys,
    "kidney": _kidneys,
    "lymph_nodes": _lymph_nodes,
    "lymph nodes": _lymph_nodes,
    "blood_vessels": _blood_vessels,
    "blood vessels": _blood_vessels,
}
