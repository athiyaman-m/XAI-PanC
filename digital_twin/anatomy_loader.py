"""
anatomy_loader.py — Load realistic anatomical 3D meshes.

Supports:
  - STL, OBJ, GLTF, GLB (via PyVista/Trimesh when available)
  - Procedural fallback using surface-biased point clouds
    for smooth alpha-shape rendering.

Designed so real DICOM/CT-derived meshes dropped in models/meshes/
load automatically — zero code changes required.
"""

import os
import numpy as np
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
MESH_DIR = os.path.join(os.path.dirname(HERE), "models", "meshes")

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


def load_organ_mesh(organ_name: str) -> dict:
    """Load or generate a 3D mesh for *organ_name*.

    Cached after first load.
    Returns dict with keys: vertices, faces, source, path.
    """
    key = organ_name.lower().strip()
    if key in _MESH_CACHE:
        return _MESH_CACHE[key]

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


# ── Procedural point-cloud generators ─────────────────────────

def _surface_ellipsoid(cx, cy, cz, rx, ry, rz, n=400):
    """Points biased toward surface for smooth alpha-shape."""
    pts = []
    for _ in range(n):
        t = np.random.uniform(0, 2 * np.pi)
        p = np.random.uniform(0, np.pi)
        r = np.random.beta(2, 1.5)  # surface bias
        pts.append([cx + r * rx * np.sin(p) * np.cos(t),
                    cy + r * ry * np.sin(p) * np.sin(t),
                    cz + r * rz * np.cos(p)])
    return pts


def _surface_tube(path_pts, radii, n_circ=14):
    """Tube with surface bias — good for pancreas/ducts."""
    pa = np.array(path_pts)
    n = len(pa)
    pts = []
    for i in range(n):
        r = radii[i] if hasattr(radii, '__len__') else radii
        if i == 0:
            d = pa[1] - pa[0]
        elif i == n - 1:
            d = pa[-1] - pa[-2]
        else:
            d = pa[i + 1] - pa[i - 1]
        dn = np.linalg.norm(d) + 1e-10
        d = d / dn
        if abs(d[0]) < 0.9:
            p1 = np.cross(d, [1, 0, 0])
        else:
            p1 = np.cross(d, [0, 1, 0])
        p1 = p1 / (np.linalg.norm(p1) + 1e-10)
        p2 = np.cross(d, p1)
        for _ in range(n_circ):
            a = np.random.uniform(0, 2 * np.pi)
            rd = r * np.random.beta(2, 1.5)
            pts.append(pa[i] + rd * (np.cos(a) * p1 + np.sin(a) * p2))
    return pts


def _deformed_surface(cx, cy, cz, rx, ry, rz, deform_fn, n=400):
    """Deformed ellipsoid with surface bias."""
    pts = []
    for _ in range(n):
        t = np.random.uniform(0, 2 * np.pi)
        p = np.random.uniform(0, np.pi)
        r = np.random.beta(2, 1.5)
        x = r * rx * np.sin(p) * np.cos(t)
        y = r * ry * np.sin(p) * np.sin(t)
        z = r * rz * np.cos(p)
        dx, dy, dz = deform_fn(x, y, z)
        pts.append([cx + x + dx, cy + y + dy, cz + z + dz])
    return pts


# ── Individual organ generators ───────────────────────────────

def _pancreas():
    n = 80
    s_ = np.linspace(0, 1, n)
    # Curved path: head (right) to tail (left, slightly up)
    cx = 0.7 - 1.7 * s_ + 0.3 * s_ ** 2
    cy = 0.0 + 0.4 * s_ - 0.6 * s_ ** 2
    cz = -0.15 + 0.55 * s_
    radii = np.zeros(n)
    for i, si in enumerate(s_):
        if si < 0.18:
            radii[i] = 0.20 * (1 + 0.30 * np.sin(si / 0.18 * np.pi))
        elif si < 0.28:
            radii[i] = 0.20 - (si - 0.18) / 0.10 * 0.10
        elif si < 0.72:
            radii[i] = 0.09
        else:
            radii[i] = 0.09 * (1 - (si - 0.72) / 0.28 * 0.50)
    return _surface_tube(list(zip(cx, cy, cz)), radii, 16)


def _liver():
    def deform(x, y, z):
        if x > 0:
            return (0.12 * z, 0.04 * y, -0.03 * x ** 2)
        else:
            return (0.12 * z - 0.18, 0.04 * y + 0.06, -0.06 * x)
    return _deformed_surface(0.55, 0.18, 0.55, 0.52, 0.22, 0.32, deform, 500)


def _gallbladder():
    pts = []
    for _ in range(250):
        t = np.random.uniform(0, 2 * np.pi)
        p = np.random.uniform(0, np.pi)
        r = np.random.beta(2, 1.5)
        s = 1 + 0.35 * np.cos(p)
        pts.append([0.78 + r * 0.07 * s * np.sin(p) * np.cos(t),
                    0.32 + r * 0.07 * s * np.sin(p) * np.sin(t),
                    0.28 + r * 0.11 * s * np.cos(p)])
    return pts


def _bile_duct():
    pa = [[0.48, 0.28, 0.32], [0.50, 0.26, 0.22],
          [0.52, 0.24, 0.12], [0.52, 0.22, 0.04],
          [0.50, 0.20, -0.02]]
    return _surface_tube(pa, np.linspace(0.035, 0.025, 5), 10)


def _duodenum():
    pa = [[0.42, 0.14, 0.22], [0.52, 0.18, 0.14], [0.56, 0.20, 0.04],
          [0.58, 0.18, -0.04], [0.56, 0.14, -0.12], [0.52, 0.08, -0.18],
          [0.46, 0.04, -0.24], [0.38, 0.0, -0.26], [0.30, -0.04, -0.24],
          [0.24, -0.06, -0.20], [0.20, -0.06, -0.14]]
    return _surface_tube(pa, np.linspace(0.07, 0.05, 11), 12)


def _stomach():
    pa = [[-0.18, 0.42, 0.45], [-0.22, 0.46, 0.36], [-0.28, 0.46, 0.26],
          [-0.32, 0.42, 0.16], [-0.36, 0.32, 0.08], [-0.32, 0.18, -0.02],
          [-0.22, 0.08, -0.10], [-0.12, 0.0, -0.16], [-0.04, -0.04, -0.20],
          [0.06, -0.04, -0.20], [0.14, 0.0, -0.16]]
    return _surface_tube(pa, np.linspace(0.22, 0.12, 11), 16)


def _spleen():
    def deform(x, y, z):
        return (0.18 * z, -0.12 * x + 0.04 * np.sin(z * 5), 0)
    return _deformed_surface(-1.02, -0.08, 0.35, 0.18, 0.10, 0.26, deform, 350)


def _lungs():
    pts = []
    for side in (-1, 1):
        for _ in range(350):
            t = np.random.uniform(0, 2 * np.pi)
            p = np.random.uniform(0, np.pi)
            r = np.random.beta(2, 1.5)
            x = side * 0.62 + r * 0.32 * np.sin(p) * np.cos(t)
            y = 0.04 + r * 0.20 * np.sin(p) * np.sin(t)
            z = 1.15 + r * 0.50 * np.cos(p)
            if side < 0 and y > 0 and z < 0.15:
                y += 0.08  # cardiac notch
            pts.append([x, y, z])
    return pts


def _heart():
    def deform(x, y, z):
        return (0.06 * x ** 2, 0.04 * z, -0.04 * y ** 2)
    return _deformed_surface(-0.10, 0.22, 1.0, 0.22, 0.16, 0.28, deform, 300)


def _kidneys():
    pts = []
    for side in (-0.55, 0.55):
        def defn(x, y, z, s=side):
            return (0.06 * z, -0.04 * x * s / 0.55, 0)
        pts.extend(_deformed_surface(side, -0.18, 0.0, 0.11, 0.06, 0.16,
                                      lambda x, y, z, s=side: defn(x, y, z, s), 200))
    return pts


def _lymph_nodes():
    sites = [[0.18, -0.12, 0.08], [-0.08, -0.08, 0.04],
             [-0.28, -0.18, 0.12], [0.48, -0.08, -0.12],
             [-0.48, -0.12, 0.18], [0.0, -0.28, -0.08],
             [0.65, 0.02, 0.55], [-0.58, 0.02, 0.75],
             [0.28, 0.08, 0.85], [-0.28, 0.08, 0.85]]
    pts = []
    for sx, sy, sz in sites:
        pts.extend(_surface_ellipsoid(sx, sy, sz, 0.03, 0.03, 0.03, 25))
    return pts


def _blood_vessels():
    pts = []
    for z in np.linspace(-1.2, 1.8, 30):
        r = 0.05 + 0.015 * (z + 1.2) / 3.0
        for _ in range(6):
            a = np.random.uniform(0, 2 * np.pi)
            rd = r * np.random.beta(2, 1.5)
            pts.append([0.0 + rd * np.cos(a), -0.32 + rd * np.sin(a), z])
    for t in np.linspace(0, 1, 20):
        x = 0.45 * t
        y = -0.08 + 0.08 * t
        z = 0.45 - 0.25 * t
        r = 0.035 * (1 - 0.3 * t)
        for _ in range(5):
            a = np.random.uniform(0, 2 * np.pi)
            rd = r * np.random.beta(2, 1.5)
            pts.append([x + rd * np.cos(a), y + rd * np.sin(a), z])
    return pts


def _procedural_organ_mesh(key: str) -> dict:
    gen = _GENERATORS.get(key)
    if gen is None:
        gen = _surface_ellipsoid  # fallback sphere
    verts = np.array(gen(), dtype=np.float64)
    return dict(vertices=verts, faces=None, source="procedural", path=None)


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
