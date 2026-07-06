"""
lesion_mapper.py — Localised tumour/lesion visualisation.

Never colour an entire organ red. Instead create localised
lesion hotspots with glow, opacity, and bounding region.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional


def create_lesion_mesh(position, size=0.08, opacity=0.7, colour="#ff3333",
                       glow=True, n_points=200):
    """
    Generate a Mesh3d trace for a localised lesion at *position*.

    Parameters
    ----------
    position : (x, y, z) tuple
    size : float — radius of the lesion
    opacity : float — 0-1
    colour : str — hex colour
    glow : bool — add a larger transparent outer halo
    n_points : int — resolution of the point cloud

    Returns
    -------
    list[go.Mesh3d]  — 1 or 2 traces (core + optional glow halo)
    """
    cx, cy, cz = position
    traces = []

    # Core lesion
    pts = []
    for _ in range(n_points):
        t = np.random.uniform(0, 2 * np.pi)
        p = np.random.uniform(0, np.pi)
        r = np.random.uniform(0, 1) ** (1 / 3) * size
        pts.append([
            cx + r * np.sin(p) * np.cos(t),
            cy + r * np.sin(p) * np.sin(t),
            cz + r * np.cos(p),
        ])
    pts = np.array(pts)
    traces.append(go.Mesh3d(
        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
        alphahull=4, opacity=opacity,
        color=colour, showscale=False,
        name="Lesion",
        lighting=dict(ambient=0.3, diffuse=0.8, specular=0.4, roughness=0.3),
        hoverinfo="text",
        text=f"<b>Lesion</b><br>Size: {size*100:.1f} mm<br>Confidence: {opacity*100:.0f}%",
    ))

    # Glow halo
    if glow:
        halo_pts = []
        for _ in range(n_points // 2):
            t = np.random.uniform(0, 2 * np.pi)
            p = np.random.uniform(0, np.pi)
            r = np.random.uniform(0, 1) ** (1 / 3) * size * 2.2
            halo_pts.append([
                cx + r * np.sin(p) * np.cos(t),
                cy + r * np.sin(p) * np.sin(t),
                cz + r * np.cos(p),
            ])
        halo_pts = np.array(halo_pts)
        traces.append(go.Mesh3d(
            x=halo_pts[:, 0], y=halo_pts[:, 1], z=halo_pts[:, 2],
            alphahull=4, opacity=opacity * 0.25,
            color=colour, showscale=False,
            name="Lesion Glow",
            lighting=dict(ambient=0.5, diffuse=0.5, specular=0.1, roughness=0.8),
            hoverinfo="skip",
        ))

    return traces


def create_metastasis_chain(positions, colour="#ff6666", opacity=0.5):
    """
    Scatter3d chain showing metastatic spread path.

    Parameters
    ----------
    positions : list of (x, y, z) tuples

    Returns
    -------
    list[go.Scatter3d]
    """
    pos = np.array(positions)
    trace = go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        mode='markers+lines',
        marker=dict(
            size=6 + np.linspace(0, 10, len(pos)),
            color=colour,
            opacity=opacity,
            symbol='diamond',
        ),
        line=dict(color=colour, width=2, dash='dot'),
        name='Metastasis',
        hoverinfo='text',
        text=[f'Metastasis #{i+1}' for i in range(len(pos))],
    )
    return [trace]
