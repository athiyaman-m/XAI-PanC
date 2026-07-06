"""
organ_renderer.py — Multi-layer 3D visualisation of the digital twin.

Supports 7 visualisation modes:
  1. Normal anatomy
  2. Risk heatmap
  3. Disease progression
  4. Blood supply
  5. Lymphatic spread
  6. Tumour localisation
  7. Treatment response simulation

All rendering uses Plotly (go.Mesh3d, go.Surface, go.Scatter3d).
When real meshes are available via PyVista they are converted to
Plotly traces; otherwise, procedural point-clouds are used.
"""

import numpy as np
import plotly.graph_objects as go
from typing import Optional

from .organ_models import Organ, Lesion
from .anatomy_loader import load_organ_mesh
from .lesion_mapper import create_lesion_mesh, create_metastasis_chain
from .organ_animation import spread_path, organ_pulse


# ── Colour scales used across modes ────────────────────────────

RISK_COLORSCALE = [
    [0.00, "#2ecc71"],
    [0.20, "#2ecc71"],
    [0.35, "#f1c40f"],
    [0.55, "#f39c12"],
    [0.75, "#e74c3c"],
    [1.00, "#c0392b"],
]

HEALTH_COLORSCALE = [
    [0.00, "#c0392b"],
    [0.25, "#e74c3c"],
    [0.45, "#f39c12"],
    [0.65, "#f1c40f"],
    [0.80, "#2ecc71"],
    [1.00, "#27ae60"],
]


def _organ_to_mesh3d(organ: Organ, colour: str = None, opacity: float = None,
                     per_vertex_colour: np.ndarray = None) -> go.Mesh3d:
    """
    Convert an Organ's mesh data into a go.Mesh3d trace.

    Falls back to alphahull (point-cloud) when no faces are available.
    """
    mesh_data = organ.plotly_mesh
    if mesh_data is None:
        mesh_data = load_organ_mesh(organ.name)
        organ.plotly_mesh = mesh_data

    verts = mesh_data.get('vertices')
    faces = mesh_data.get('faces')
    c = colour or organ.colour
    op = opacity if opacity is not None else organ.opacity

    kwargs = dict(
        opacity=op,
        showscale=False,
        lighting=dict(ambient=0.40, diffuse=0.78, specular=0.25, roughness=0.35),
        hoverinfo='text',
        text=f"<b>{organ.name}</b>",
        name=organ.name,
    )

    if faces is not None and len(faces) > 0:
        # Real mesh with faces
        kwargs.update(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                      i=faces[:, 0], j=faces[:, 1], k=faces[:, 2])
        if per_vertex_colour is not None:
            kwargs['vertexcolor'] = per_vertex_colour
        else:
            kwargs['color'] = c
    else:
        # Point-cloud → alpha shape
        kwargs.update(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                      alphahull=5)
        if per_vertex_colour is not None:
            kwargs['vertexcolor'] = per_vertex_colour
        else:
            kwargs['color'] = c

    return go.Mesh3d(**kwargs)


# ── Mode 1: Normal Anatomy ────────────────────────────────────

def render_normal(organs: dict[str, Organ]) -> list:
    """Standard anatomical view — healthy colours with subtle gradient."""
    traces = [_torso_outline()]
    for name, organ in organs.items():
        t = _organ_to_mesh3d(organ, colour="#4a7a9a", opacity=0.70)
        t.text = f"<b>{name}</b><br><i>Normal anatomy</i>"
        traces.append(t)
    return traces


# ── Mode 2: Risk Heatmap (default) ────────────────────────────

def render_risk_heatmap(organs: dict[str, Organ]) -> list:
    """Risk-based colouring — green → yellow → orange → red."""
    traces = [_torso_outline()]
    for name, organ in organs.items():
        t = _organ_to_mesh3d(organ, colour=organ.colour, opacity=organ.opacity)
        t.text = (
            f"<b>{name}</b><br>"
            f"Health: {organ.health_score:.0f}%<br>"
            f"Risk: {organ.risk_score:.0f}%<br>"
            f"Stage: {organ.disease_stage}"
        )
        traces.append(t)

        # Lesions
        for les in organ.lesions:
            traces.extend(create_lesion_mesh(
                les.position, les.size, les.opacity,
                "#ff3333", glow=les.glow_intensity > 0.3,
            ))

        # Pulsing glow for high-risk organs
        if organ.risk_score >= 50:
            centre = _organ_centre(organ)
            traces.extend(organ_pulse(centre, organ.risk_score, organ.colour))

    # Metastasis chains
    for name, organ in organs.items():
        if organ.spread_regions and organ.risk_score > 40:
            src = _organ_centre(organ)
            for target_name in organ.spread_regions:
                if target_name in organs:
                    tgt = _organ_centre(organs[target_name])
                    traces.extend(spread_path(src, tgt, colour="#ff6666"))

    return traces


# ── Mode 3: Disease Progression ───────────────────────────────

def render_disease_progression(organs: dict[str, Organ]) -> list:
    """Similar to risk heatmap but with explicit tumour markers + spread."""
    traces = [_torso_outline()]
    for name, organ in organs.items():
        t = _organ_to_mesh3d(organ, colour=organ.colour, opacity=organ.opacity)
        t.text = (
            f"<b>{name}</b><br>"
            f"Tumour: {organ.tumour_size:.2f} cm<br>"
            f"Risk: {organ.risk_score:.0f}%<br>"
            f"Spread: {', '.join(organ.spread_regions) or 'None'}"
        )
        traces.append(t)

        for les in organ.lesions:
            traces.extend(create_lesion_mesh(
                les.position, les.size, les.opacity, "#ff3333", glow=True,
            ))

    # Spread arrows
    for name, organ in organs.items():
        if organ.spread_regions:
            src = _organ_centre(organ)
            for t_name in organ.spread_regions:
                if t_name in organs:
                    tgt = _organ_centre(organs[t_name])
                    traces.extend(spread_path(src, tgt, "#ff8844", n_particles=15))
    return traces


# ── Mode 4: Blood Supply ──────────────────────────────────────

def render_blood_supply(organs: dict[str, Organ]) -> list:
    """Vascular overlay — semi-transparent organs + vessel traces."""
    traces = [_torso_outline()]
    for name, organ in organs.items():
        t = _organ_to_mesh3d(organ, colour="#4466aa", opacity=0.35)
        traces.append(t)

    # Blood vessel mesh
    vessel_data = _load_vessel_mesh()
    if vessel_data is not None:
        verts = vessel_data['vertices']
        traces.append(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            alphahull=4, opacity=0.6,
            color="#cc3333", showscale=False,
            lighting=dict(ambient=0.5, diffuse=0.6, specular=0.1, roughness=0.6),
            name="Blood Vessels",
            hoverinfo='text', text="<b>Vascular network</b>",
        ))
    return traces


# ── Mode 5: Lymphatic Spread ──────────────────────────────────

def render_lymphatic_spread(organs: dict[str, Organ]) -> list:
    """Lymph nodes highlighted + spread paths to organs."""
    traces = [_torso_outline()]
    for name, organ in organs.items():
        if name == "Lymph Nodes":
            continue
        t = _organ_to_mesh3d(organ, colour="#667788", opacity=0.40)
        traces.append(t)

    lymph = organs.get("Lymph Nodes")
    if lymph:
        t = _organ_to_mesh3d(lymph, colour="#ffaa44", opacity=0.85)
        traces.append(t)

        if lymph.spread_regions:
            src = _organ_centre(lymph)
            for t_name in lymph.spread_regions:
                if t_name in organs:
                    tgt = _organ_centre(organs[t_name])
                    traces.extend(spread_path(src, tgt, "#ffaa44", n_particles=25))
    return traces


# ── Mode 6: Tumour Localisation ───────────────────────────────

def render_tumour_localisation(organs: dict[str, Organ]) -> list:
    """Focus on tumour sites — dim healthy tissue, highlight lesions."""
    traces = [_torso_outline()]
    for name, organ in organs.items():
        has_tumour = organ.tumour_size > 0.01 or len(organ.lesions) > 0
        t = _organ_to_mesh3d(
            organ,
            colour=organ.colour if has_tumour else "#334455",
            opacity=0.85 if has_tumour else 0.20,
        )
        traces.append(t)

        for les in organ.lesions:
            traces.extend(create_lesion_mesh(
                les.position, les.size * 1.3, min(1.0, les.opacity + 0.15),
                "#ff2222", glow=True,
            ))

        # Tumour bounding box (wireframe)
        if organ.tumour_size > 0.1:
            centre = _organ_centre(organ)
            traces.extend(_bounding_box(centre, organ.tumour_size * 0.5, organ.colour))
    return traces


# ── Mode 7: Treatment Response Simulation ─────────────────────

def render_treatment_simulation(organs: dict[str, Organ], treatment_name: str = "",
                                response_prob: float = 0.0) -> list:
    """Simulated post-treatment state — greener = better response."""
    traces = [_torso_outline()]
    for name, organ in organs.items():
        # Simulate response: health improves proportionally to response_prob
        simulated_health = min(100, organ.health_score + response_prob * 30)
        simulated_risk = max(0, organ.risk_score - response_prob * 25)

        t = _organ_to_mesh3d(organ, colour=_health_to_colour(simulated_health), opacity=0.75)
        t.text = (
            f"<b>{name}</b><br>"
            f"Projected Health: {simulated_health:.0f}%<br>"
            f"Projected Risk: {simulated_risk:.0f}%<br>"
            f"Treatment: {treatment_name or 'None'}"
        )
        traces.append(t)
    return traces


# ── Helpers ───────────────────────────────────────────────────

def _torso_outline() -> go.Surface:
    """Semi-transparent torso silhouette."""
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(-1, 1, 80)
    U, V = np.meshgrid(u, v)
    w = 1.5 * (1 - 0.12 * V ** 2 + 0.04 * V ** 4)
    d = 0.75 * (1 - 0.10 * V ** 2 + 0.02 * V ** 4)
    X_t = w * np.cos(U)
    Y_t = d * np.sin(U)
    Z_t = 2.2 * V
    shoulder = Z_t > 1.6
    X_t[shoulder] *= 1 + 0.15 * (Z_t[shoulder] - 1.6) / 0.6

    return go.Surface(
        x=X_t, y=Y_t, z=Z_t,
        opacity=0.03,
        colorscale=[[0, '#3b82f6'], [1, '#3b82f6']],
        showscale=False,
        contours=dict(
            x=dict(show=True, color='rgba(59,130,246,0.06)', width=1),
            y=dict(show=True, color='rgba(59,130,246,0.06)', width=1),
            z=dict(show=False),
        ),
        hoverinfo='skip',
    )


def _organ_centre(organ: Organ) -> tuple:
    """Estimate organ centre from mesh vertices."""
    md = organ.plotly_mesh
    if md is None:
        md = load_organ_mesh(organ.name)
        organ.plotly_mesh = md
    verts = md['vertices']
    if len(verts) == 0:
        return (0, 0, 0)
    return tuple(np.mean(verts, axis=0))


def _load_vessel_mesh():
    """Load or generate blood vessel mesh."""
    from .anatomy_loader import load_organ_mesh
    return load_organ_mesh("blood_vessels")


def _bounding_box(centre: tuple, half_size: float, colour: str) -> list:
    """Wireframe bounding box around a tumour."""
    cx, cy, cz = centre
    hs = half_size
    corners = np.array([
        [cx - hs, cy - hs, cz - hs], [cx + hs, cy - hs, cz - hs],
        [cx + hs, cy + hs, cz - hs], [cx - hs, cy + hs, cz - hs],
        [cx - hs, cy - hs, cz + hs], [cx + hs, cy - hs, cz + hs],
        [cx + hs, cy + hs, cz + hs], [cx - hs, cy + hs, cz + hs],
    ])
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    traces = []
    for i, j in edges:
        traces.append(go.Scatter3d(
            x=[corners[i, 0], corners[j, 0]],
            y=[corners[i, 1], corners[j, 1]],
            z=[corners[i, 2], corners[j, 2]],
            mode='lines',
            line=dict(color=colour, width=2, dash='dash'),
            hoverinfo='skip',
            showlegend=False,
        ))
    return traces


def _health_to_colour(health: float) -> str:
    if health >= 70:
        return "#2ecc71"
    if health >= 45:
        return "#f1c40f"
    if health >= 20:
        return "#f39c12"
    return "#e74c3c"


# ── Renderer registry ─────────────────────────────────────────

RENDER_MODES = {
    "Normal Anatomy": render_normal,
    "Risk Heatmap": render_risk_heatmap,
    "Disease Progression": render_disease_progression,
    "Blood Supply": render_blood_supply,
    "Lymphatic Spread": render_lymphatic_spread,
    "Tumour Localisation": render_tumour_localisation,
    "Treatment Response": render_treatment_simulation,
}
