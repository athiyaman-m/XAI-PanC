import numpy as np
from typing import Callable, Optional

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def _surface_noise(u: np.ndarray, v: np.ndarray, freq: float = 4.0,
                   amp: float = 0.02, seed: int = 42) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = np.zeros_like(u)
    for octave in range(3):
        f = freq * (2 ** octave)
        a = amp / (octave + 1)
        phase_u = rng.uniform(0, 2 * np.pi)
        phase_v = rng.uniform(0, 2 * np.pi)
        noise += a * (np.sin(u * f + phase_u) * np.cos(v * f * 0.7 + phase_v)
                      + np.sin(u * f * 1.3 + phase_u * 1.5) * 0.5)
    return noise


def ellipsoid_mesh(
    cx, cy, cz, rx, ry, rz,
    deform_fn: Optional[Callable] = None,
    n_u: int = 48, n_v: int = 32,
    noise_amp: float = 0.0, noise_freq: float = 4.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    v = np.linspace(0, np.pi, n_v)
    U, V = np.meshgrid(u, v, indexing='ij')

    X = cx + rx * np.sin(V) * np.cos(U)
    Y = cy + ry * np.sin(V) * np.sin(U)
    Z = cz + rz * np.cos(V)

    if deform_fn:
        for i in range(n_u):
            for j in range(n_v):
                dx, dy, dz = deform_fn(X[i, j] - cx, Y[i, j] - cy, Z[i, j] - cz)
                if dx or dy or dz:
                    X[i, j] += dx
                    Y[i, j] += dy
                    Z[i, j] += dz

    if noise_amp > 0:
        noise_val = _surface_noise(U, V, freq=noise_freq, amp=noise_amp, seed=seed)
        for i in range(n_u):
            for j in range(n_v):
                nx = X[i, j] - cx
                ny = Y[i, j] - cy
                nz = Z[i, j] - cz
                norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-10
                nx /= norm
                ny /= norm
                nz /= norm
                displacement = noise_val[i, j]
                X[i, j] += nx * displacement
                Y[i, j] += ny * displacement
                Z[i, j] += nz * displacement

    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    tris = []
    for a in range(n_u - 1):
        for b in range(n_v - 1):
            p1 = a * n_v + b
            p2 = (a + 1) * n_v + b
            p3 = (a + 1) * n_v + (b + 1)
            p4 = a * n_v + (b + 1)
            tris.append([p1, p2, p3])
            tris.append([p1, p3, p4])

    return verts, np.array(tris, dtype=np.int32)


def tube_mesh(
    path_pts: list,
    radii,
    n_circ: int = 24,
    noise_amp: float = 0.0, noise_freq: float = 5.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    pa = np.array(path_pts)
    n = len(pa)
    verts = []

    rng_noise = np.random.RandomState(seed)

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
        for j in range(n_circ):
            a = 2 * np.pi * j / n_circ
            base_r = r
            if noise_amp > 0:
                noise_r = base_r + noise_amp * (
                    np.sin(a * noise_freq + i * 0.5) * 0.5
                    + np.sin(a * noise_freq * 2.3 + i * 1.7 + rng_noise.rand() * 0.5) * 0.3
                )
            else:
                noise_r = base_r
            verts.append(pa[i] + noise_r * (np.cos(a) * p1 + np.sin(a) * p2))

    verts = np.array(verts)
    tris = []
    for a in range(n - 1):
        for b in range(n_circ):
            p1 = a * n_circ + b
            p2 = (a + 1) * n_circ + b
            p3 = (a + 1) * n_circ + (b + 1) % n_circ
            p4 = a * n_circ + (b + 1) % n_circ
            tris.append([p1, p2, p3])
            tris.append([p1, p3, p4])

    return verts, np.array(tris, dtype=np.int32)


def merge_meshes(meshes: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    all_verts = []
    all_tris = []
    offset = 0
    for verts, tris in meshes:
        all_verts.append(verts)
        all_tris.append(tris + offset if offset else tris)
        offset += len(verts)
    return np.vstack(all_verts) if all_verts else np.empty((0, 3)), np.vstack(all_tris) if all_tris else np.empty((0, 3), dtype=np.int32)


def compute_vertex_normals(verts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(verts)
    for tri in faces:
        v0, v1, v2 = verts[tri[0]], verts[tri[1]], verts[tri[2]]
        n = np.cross(v1 - v0, v2 - v0)
        n_len = np.linalg.norm(n)
        if n_len > 1e-10:
            n = n / n_len
            normals[tri[0]] += n
            normals[tri[1]] += n
            normals[tri[2]] += n
    n_len = np.linalg.norm(normals, axis=1, keepdims=True)
    n_len[n_len < 1e-10] = 1
    return normals / n_len


def decimate_mesh(verts: np.ndarray, faces: np.ndarray,
                  target_verts: int = 40000) -> tuple[np.ndarray, np.ndarray]:
    """Reduce triangle count via uniform vertex clustering.

    Clusters vertices into a uniform grid (cell size derived from surface area
    so the number of occupied cells lands near `target_verts`), keeps the mean
    position of each cluster and remaps / de-duplicates faces.
    """
    n = len(verts)
    if n <= target_verts:
        return np.ascontiguousarray(verts), np.ascontiguousarray(faces)
    tri = verts[faces]
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1).sum()
    bmin = verts.min(0)
    bbox = verts.max(0) - bmin
    bbox[bbox < 1e-9] = 1e-9
    cell = float(np.sqrt(area / target_verts)) if area > 0 else float(np.cbrt(np.prod(bbox) / target_verts))
    cell = max(cell, 1e-6)
    idx = np.floor((verts - bmin) / cell).astype(np.int64)
    _, inverse = np.unique(idx, axis=0, return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    rep = np.zeros((n_clusters, 3), dtype=np.float64)
    np.add.at(rep, inverse, verts)
    counts = np.bincount(inverse).astype(np.float64)
    counts[counts < 1] = 1
    rep /= counts[:, None]

    f_new = inverse[faces]
    f_sorted = np.sort(f_new, axis=1)
    _, keep = np.unique(f_sorted, axis=0, return_index=True)
    f_keep = f_new[keep]
    nondeg = (f_keep[:, 0] != f_keep[:, 1]) & (f_keep[:, 1] != f_keep[:, 2])
    f_keep = f_keep[nondeg]
    if len(f_keep) == 0:
        return rep, np.empty((0, 3), dtype=np.int32)
    uniq, compact = np.unique(f_keep, return_inverse=True)
    return rep[uniq].astype(np.float64), compact.reshape(-1, 3).astype(np.int32)


def laplacian_smooth(verts: np.ndarray, faces: np.ndarray,
                     iterations: int = 8, factor: float = 0.5) -> np.ndarray:
    """Uniform Laplacian smoothing (taubin-style damped) preserving shape."""
    v = np.array(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    n = len(v)
    rows = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    cols = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    data = np.ones(len(rows))
    adj = csr_matrix((data, (rows, cols)), shape=(n, n))
    deg = np.array(adj.sum(1)).ravel()
    deg[deg < 1] = 1
    dinv = csr_matrix((1.0 / deg, (np.arange(n), np.arange(n))), shape=(n, n))
    lap = (dinv @ adj).tocsr()
    for _ in range(int(iterations)):
        v = v + factor * (lap @ v - v)
    return v


def keep_largest_component(verts: np.ndarray, faces: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the largest connected triangle component (removes printer
    stands, text and stray parts from downloaded STL models). Works even for
    un-indexed STL files by matching vertex positions."""
    v = np.asarray(verts, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if len(f) == 0:
        return np.ascontiguousarray(v), np.ascontiguousarray(f)
    # quantize positions -> shared vertex ids (handles un-indexed STLs)
    _, vids = np.unique(np.round(v, 5), axis=0, return_inverse=True)
    f_q = vids[f]  # (m,3) quantized vertex ids per face
    m = len(f)
    # face-adjacency: faces sharing >= 2 vertex ids are connected
    rows = np.concatenate([np.arange(m), np.arange(m), np.arange(m)])
    adj = csr_matrix((np.ones(len(rows)), (rows, f_q.ravel())),
                     shape=(m, int(vids.max()) + 1))
    adj_face = (adj @ adj.T).tocsr()
    adj_face.data = (adj_face.data >= 2).astype(np.int8)
    n_comp, labels = connected_components(adj_face, directed=False)
    if n_comp <= 1:
        return np.ascontiguousarray(v), np.ascontiguousarray(f)
    sizes = np.bincount(labels)
    largest = int(sizes.argmax())
    keep = np.flatnonzero(labels == largest)
    keep_set = set(f[keep].ravel().tolist())
    vmap = {old: new for new, old in enumerate(sorted(keep_set))}
    new_faces = np.array([[vmap[old] for old in tri] for tri in f[keep]],
                         dtype=np.int32)
    new_verts = v[list(keep_set)]
    return np.ascontiguousarray(new_verts), np.ascontiguousarray(new_faces)


def normalize_mesh(verts: np.ndarray) -> np.ndarray:
    """Translate to origin and scale so the longest axis fits in [-0.5, 0.5]."""
    v = np.asarray(verts, dtype=np.float64)
    if len(v) == 0:
        return v
    c = (v.max(0) + v.min(0)) / 2.0
    size = float(np.max(v.max(0) - v.min(0)))
    if size < 1e-9:
        return v - c
    return (v - c) / size


def place_mesh(verts: np.ndarray, centre, scale: float = 1.0) -> np.ndarray:
    """Normalize a mesh and place it at `centre` (x,y,z) with given `scale`."""
    v = normalize_mesh(verts)
    return v * scale + np.asarray(centre, dtype=np.float64)


def mirror_mesh(verts: np.ndarray, axis: int = 0) -> np.ndarray:
    """Mirror a mesh across a plane through its centroid along `axis`."""
    v = np.array(verts, dtype=np.float64)
    v[:, axis] = 2 * v[:, axis].mean() - v[:, axis]
    return v


def center_mesh(verts: np.ndarray) -> np.ndarray:
    """Translate so centroid (b-box centre) is at the origin."""
    v = np.asarray(verts, dtype=np.float64)
    if len(v) == 0:
        return v
    return v - (v.max(0) + v.min(0)) / 2.0
