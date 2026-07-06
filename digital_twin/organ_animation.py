"""
organ_animation.py — Animations for disease spread, blood flow,
and progression effects.

Each function returns a list of Plotly traces that can be added
to the scene.  They are designed to be re-created on each render
call, keeping the animation stateless and deterministic.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional


def spread_path(source: tuple, target: tuple,
                colour: str = "#ff4444", n_particles: int = 20,
                line_width: int = 3, opacity: float = 0.6):
    """
    Animated spread path with particles from *source* to *target*.

    Returns a list of Plotly traces:
      - one line trace (the path)
      - one scatter trace (moving particles along the path)
    """
    src = np.array(source)
    tgt = np.array(target)
    path_vec = tgt - src

    # Path line
    line_trace = go.Scatter3d(
        x=[src[0], tgt[0]], y=[src[1], tgt[1]], z=[src[2], tgt[2]],
        mode='lines',
        line=dict(color=colour, width=line_width, dash='dot'),
        name='Spread path',
        hoverinfo='skip',
        showlegend=False,
    )

    # Particles along the path (evenly spaced)
    fractions = np.linspace(0.05, 0.95, n_particles)
    px = src[0] + path_vec[0] * fractions
    py = src[1] + path_vec[1] * fractions
    pz = src[2] + path_vec[2] * fractions
    sizes = 2 + 6 * (1 - np.abs(fractions - 0.5) * 2)  # larger in middle

    particle_trace = go.Scatter3d(
        x=px, y=py, z=pz,
        mode='markers',
        marker=dict(
            size=sizes,
            color=colour,
            opacity=opacity,
            symbol='circle',
            line=dict(width=0),
        ),
        name='Particles',
        hoverinfo='skip',
        showlegend=False,
    )

    return [line_trace, particle_trace]


def animated_glow(centre: tuple, radius: float = 0.3,
                  colour: str = "#ff4444", n_layers: int = 3):
    """
    Concentric glowing rings around *centre*.

    Returns a list of Mesh3d traces with decreasing opacity.
    """
    traces = []
    cx, cy, cz = centre
    for i in range(n_layers):
        r = radius * (1 + i * 0.6)
        op = 0.15 / (i + 1)
        pts = []
        for _ in range(150):
            t = np.random.uniform(0, 2 * np.pi)
            p = np.random.uniform(0, np.pi)
            rad = np.random.uniform(r * 0.85, r * 1.15)
            pts.append([
                cx + rad * np.sin(p) * np.cos(t),
                cy + rad * np.sin(p) * np.sin(t),
                cz + rad * np.cos(p),
            ])
        pts = np.array(pts)
        traces.append(go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            alphahull=4, opacity=op,
            color=colour, showscale=False,
            lighting=dict(ambient=0.6, diffuse=0.4, specular=0.1, roughness=0.9),
            hoverinfo='skip',
        ))
    return traces


def organ_pulse(organ_centre: tuple, organ_risk: float,
                colour: str = "#e74c3c"):
    """
    Pulsing ring around an organ based on risk level.

    Higher risk = larger, more opaque pulse.
    """
    if organ_risk < 30:
        return []

    intensity = organ_risk / 100.0
    radius = 0.15 + intensity * 0.35
    return animated_glow(organ_centre, radius, colour, n_layers=2)
