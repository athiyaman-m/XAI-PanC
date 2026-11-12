"""Lightweight mesh file I/O (STL binary/ascii, OBJ) using only numpy.

Used to load real anatomical meshes (e.g. NIH 3D Print Exchange / Visible Human,
BodyParts3D) without requiring heavy 3D libraries such as trimesh or pyvista.
"""
from __future__ import annotations

import os
import struct
import numpy as np


def load_stl(path: str) -> dict | None:
    """Load a binary or ASCII STL file.

    Returns dict(vertices=(N,3), faces=(M,3)) in float64/int32, or None on failure.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(5)
    except OSError:
        return None
    if head[:5] == b"solid":
        return _load_stl_ascii(path)
    return _load_stl_binary(path)


def _load_stl_binary(path: str) -> dict | None:
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    if len(data) < 84:
        return None
    n_tri = struct.unpack("<I", data[80:84])[0]
    expected = 84 + n_tri * 50
    if len(data) < expected:
        return None
    tri_buf = np.frombuffer(data, dtype=np.uint8, offset=84, count=n_tri * 50)
    tri_buf = tri_buf.reshape(n_tri, 50)
    # each triangle: 12 floats (normal + 3 verts) + 2 attr bytes
    floats = np.ascontiguousarray(tri_buf[:, :48]).view(np.float32)
    v = floats.reshape(n_tri, 4, 3)[:, 1:, :].reshape(n_tri * 3, 3)
    faces = np.arange(n_tri * 3, dtype=np.int32).reshape(n_tri, 3)
    return {"vertices": v.astype(np.float64), "faces": faces, "source": "stl", "path": path}


def _load_stl_ascii(path: str) -> dict | None:
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return None
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    tri_verts: list[list[int]] = []
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "vertex" and len(parts) >= 4:
            tri_verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(tri_verts) == 3:
                idx = [len(verts), len(verts) + 1, len(verts) + 2]
                verts.extend(tri_verts)
                faces.append(idx)
                tri_verts = []
    if not faces:
        return None
    return {
        "vertices": np.asarray(verts, dtype=np.float64),
        "faces": np.asarray(faces, dtype=np.int32),
        "source": "stl", "path": path,
    }


def load_obj(path: str) -> dict | None:
    """Load a Wavefront OBJ file (vertices + faces only)."""
    try:
        with open(path, "r", errors="replace") as fh:
            lines = fh.readlines()
    except OSError:
        return None
    verts: list[list[float]] = []
    faces: list[list[int]] = []
    for line in lines:
        if line.startswith("v "):
            parts = line.split()
            verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("f "):
            idx = []
            for tok in line.split()[1:]:
                idx.append(int(tok.split("/")[0]) - 1)
            if len(idx) >= 3:
                for i in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[i], idx[i + 1]])
    if not verts or not faces:
        return None
    return {
        "vertices": np.asarray(verts, dtype=np.float64),
        "faces": np.asarray(faces, dtype=np.int32),
        "source": "obj", "path": path,
    }


def load_mesh_file(path: str) -> dict | None:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".stl":
        return load_stl(path)
    if ext == ".obj":
        return load_obj(path)
    return None


def write_stl_ascii(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write an ASCII STL file (for human-readable exports / 3D printing)."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int32)
    with open(path, "w") as fh:
        fh.write("solid mesh\n")
        for tri in f:
            fh.write("  facet normal 0 0 0\n    outer loop\n")
            for i in tri:
                fh.write(f"      vertex {v[i,0]:.6f} {v[i,1]:.6f} {v[i,2]:.6f}\n")
            fh.write("    endloop\n  endfacet\n")
        fh.write("endsolid mesh\n")
