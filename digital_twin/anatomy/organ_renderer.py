import numpy as np
import plotly.graph_objects as go

from .organ_models import Organ
from .anatomy_loader import load_organ_mesh
from .lesion_mapper import create_lesion_mesh, create_metastasis_chain
from ..visualization.organ_animation import spread_path, organ_pulse
from .mesh_utils import compute_vertex_normals


ORGAN_CENTRES = {
    "Pancreas":     (0.0, -0.10, 0.05),
    "Liver":        (0.55, 0.18, 0.55),
    "Gallbladder":  (0.78, 0.32, 0.28),
    "Bile Duct":    (0.50, 0.24, 0.12),
    "Duodenum":     (0.38, 0.04, -0.04),
    "Stomach":      (-0.26, 0.26, 0.12),
    "Spleen":       (-1.02, -0.08, 0.35),
    "Lungs":        (0.0, 0.04, 1.15),
    "Lymph Nodes":  (0.0, -0.08, 0.08),
    "Heart":        (-0.10, 0.22, 1.00),
    "Kidneys":      (0.0, -0.18, 0.0),
}

REALISTIC_ORGAN_COLORS = {
    "Pancreas":     "#c49a6c",
    "Liver":        "#7a3520",
    "Gallbladder":  "#5a7a5a",
    "Bile Duct":    "#b89878",
    "Duodenum":     "#b8856a",
    "Stomach":      "#a86858",
    "Spleen":       "#7a4e35",
    "Lungs":        "#b88878",
    "Lymph Nodes":  "#9a7860",
    "Heart":        "#7a3030",
    "Kidneys":      "#5a4e3a",
}

RISK_COLORSCALE = [
    [0.00, "#34d399"], [0.20, "#34d399"],
    [0.35, "#fbbf24"], [0.55, "#fb923c"],
    [0.75, "#f87171"], [1.00, "#dc2626"],
]

HEALTH_COLORSCALE = [
    [0.00, "#dc2626"], [0.25, "#f87171"],
    [0.45, "#fb923c"], [0.65, "#fbbf24"],
    [0.80, "#34d399"], [1.00, "#059669"],
]


def _hex_to_rgb(hex_color: str) -> tuple:
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def _rgb_to_hex(r, g, b):
    return f"#{int(r):02x}{int(g):02x}{int(b):02x}"


def _vary_color(hex_color: str, amount: float = 0.08, seed: int = 0) -> np.ndarray:
    r, g, b = _hex_to_rgb(hex_color)
    rng = np.random.RandomState(seed)
    r_var = r + rng.randint(-int(r * amount), int(r * amount) + 1)
    g_var = g + rng.randint(-int(g * amount), int(g * amount) + 1)
    b_var = b + rng.randint(-int(b * amount), int(b * amount) + 1)
    return np.clip(r_var, 0, 255), np.clip(g_var, 0, 255), np.clip(b_var, 0, 255)


def _per_vertex_color(base_hex: str, verts: np.ndarray, noise_amp: float = 12,
                      vary_lighting: bool = True) -> np.ndarray:
    r0, g0, b0 = _hex_to_rgb(base_hex)
    n_verts = len(verts)
    rng = np.random.RandomState(42)
    r_vals = np.full(n_verts, r0, dtype=np.float64)
    g_vals = np.full(n_verts, g0, dtype=np.float64)
    b_vals = np.full(n_verts, b0, dtype=np.float64)
    if vary_lighting:
        # Sub-surface scattering simulation
        z_norm = (verts[:, 2] - verts[:, 2].min()) / (verts[:, 2].max() - verts[:, 2].min() + 1e-10)
        # Top-down lighting gradient
        height_light = (z_norm - 0.3) * 30
        r_vals = np.clip(r_vals + height_light, 0, 255)
        g_vals = np.clip(g_vals + height_light * 0.6, 0, 255)
        b_vals = np.clip(b_vals + height_light * 0.3, 0, 255)
        # Lateral shading for depth
        x_norm = (verts[:, 0] - verts[:, 0].min()) / (verts[:, 0].max() - verts[:, 0].min() + 1e-10)
        lateral_shade = (x_norm - 0.5) * 15
        r_vals = np.clip(r_vals + lateral_shade, 0, 255)
        g_vals = np.clip(g_vals + lateral_shade * 0.7, 0, 255)
        b_vals = np.clip(b_vals + lateral_shade * 0.5, 0, 255)
        # Fine noise for tissue texture
        noise = rng.randn(n_verts) * noise_amp * 0.7
        r_vals = np.clip(r_vals + noise, 0, 255)
        g_vals = np.clip(g_vals + noise * 0.85, 0, 255)
        b_vals = np.clip(b_vals + noise * 0.7, 0, 255)
        # Vascular tint (red shift at depth)
        depth_tint = np.clip((0.5 - z_norm) * 12, 0, 10)
        r_vals = np.clip(r_vals + depth_tint * 2, 0, 255)
    colors = np.column_stack([r_vals, g_vals, b_vals]) / 255.0
    return colors


def _organ_to_mesh3d(organ: Organ, colour: str = None, opacity: float = None,
                     per_vertex_colour: np.ndarray = None,
                     realistic: bool = True) -> go.Mesh3d:
    mesh_data = organ.plotly_mesh
    if mesh_data is None:
        mesh_data = load_organ_mesh(organ.name)
        organ.plotly_mesh = mesh_data
    verts = mesh_data.get('vertices')
    faces = mesh_data.get('faces')
    c = colour or organ.colour
    op = opacity if opacity is not None else organ.opacity
    if realistic and colour is None and per_vertex_colour is None:
        base_color = REALISTIC_ORGAN_COLORS.get(organ.name, c)
        per_vertex_colour = _per_vertex_color(base_color, verts, noise_amp=10)
    kwargs = dict(
        opacity=op, showscale=False,
        lighting=dict(ambient=0.40, diffuse=0.85, specular=0.30, roughness=0.35, fresnel=0.20),
        lightposition=dict(x=0, y=-3, z=5),
        hoverinfo='text', name=organ.name,
    )
    if faces is not None and len(faces) > 0:
        kwargs.update(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                      i=faces[:, 0], j=faces[:, 1], k=faces[:, 2])
        if per_vertex_colour is not None:
            kwargs['vertexcolor'] = per_vertex_colour
        else:
            kwargs['color'] = c
    else:
        kwargs.update(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], alphahull=5)
        if per_vertex_colour is not None:
            kwargs['vertexcolor'] = per_vertex_colour
        else:
            kwargs['color'] = c
    return go.Mesh3d(**kwargs)


def render_normal(organs: dict[str, Organ]) -> list:
    traces = [_torso_outline()]
    for name, organ in organs.items():
        t = _organ_to_mesh3d(organ, realistic=True)
        t.text = f"<b>{name}</b><br><i>Normal anatomy</i>"
        traces.append(t)
    return traces


def _pancreatic_tumor_traces(organs) -> list:
    from .real_ct_twin import _tumor_trace
    return _tumor_trace(organs)


def render_risk_heatmap(organs: dict[str, Organ]) -> list:
    traces = [_torso_outline()]
    for name, organ in organs.items():
        if organ.risk_score > 30:
            pv_color = _per_vertex_color(organ.colour, organ.plotly_mesh['vertices'],
                                         noise_amp=8, vary_lighting=True)
            t = _organ_to_mesh3d(organ, per_vertex_colour=pv_color, opacity=organ.opacity)
        else:
            base = REALISTIC_ORGAN_COLORS.get(name, organ.colour)
            pv_color = _per_vertex_color(base, organ.plotly_mesh['vertices'],
                                         noise_amp=10, vary_lighting=True)
            t = _organ_to_mesh3d(organ, per_vertex_colour=pv_color, opacity=organ.opacity)
        t.text = (f"<b>{name}</b><br>Health: {organ.health_score:.0f}%<br>"
                  f"Risk: {organ.risk_score:.0f}%<br>Stage: {organ.disease_stage}")
        traces.append(t)
        for les in organ.lesions:
            traces.extend(create_lesion_mesh(les.position, les.size, les.opacity, "#ff3333",
                                             glow=les.glow_intensity > 0.3,
                                             layers=les.layers, seed=les.seed))
        if organ.risk_score >= 50:
            centre = _organ_centre(organ)
            traces.extend(organ_pulse(centre, organ.risk_score, organ.colour))
    for name, organ in organs.items():
        if organ.spread_regions and organ.risk_score > 40:
            src = _organ_centre(organ)
            for target_name in organ.spread_regions:
                if target_name in organs:
                    tgt = _organ_centre(organs[target_name])
                    traces.extend(spread_path(src, tgt, colour="#ff6666"))
    traces.extend(_pancreatic_tumor_traces(organs))
    return traces


def render_disease_progression(organs: dict[str, Organ]) -> list:
    traces = [_torso_outline()]
    for name, organ in organs.items():
        base = REALISTIC_ORGAN_COLORS.get(name, organ.colour)
        pv_color = _per_vertex_color(base, organ.plotly_mesh['vertices'],
                                     noise_amp=8, vary_lighting=True)
        t = _organ_to_mesh3d(organ, per_vertex_colour=pv_color, opacity=organ.opacity)
        t.text = (f"<b>{name}</b><br>Tumour: {organ.tumour_size:.2f} cm<br>"
                  f"Risk: {organ.risk_score:.0f}%<br>Spread: {', '.join(organ.spread_regions) or 'None'}")
        traces.append(t)
        for les in organ.lesions:
            traces.extend(create_lesion_mesh(les.position, les.size, les.opacity, "#ff3333",
                                             glow=True, layers=les.layers, seed=les.seed))
    for name, organ in organs.items():
        if organ.spread_regions:
            src = _organ_centre(organ)
            for t_name in organ.spread_regions:
                if t_name in organs:
                    tgt = _organ_centre(organs[t_name])
                    traces.extend(spread_path(src, tgt, "#ff8844", n_particles=12))
    traces.extend(_pancreatic_tumor_traces(organs))
    return traces


def render_blood_supply(organs: dict[str, Organ]) -> list:
    traces = [_torso_outline()]
    for name, organ in organs.items():
        base = REALISTIC_ORGAN_COLORS.get(name, "#4466aa")
        pv_color = _per_vertex_color(base, organ.plotly_mesh['vertices'],
                                     noise_amp=6, vary_lighting=True)
        t = _organ_to_mesh3d(organ, per_vertex_colour=pv_color, opacity=0.40)
        traces.append(t)
    vessel_data = _load_vessel_mesh()
    if vessel_data is not None:
        verts = vessel_data['vertices']
        traces.append(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            alphahull=4, opacity=0.7, color="#cc3333", showscale=False,
            lighting=dict(ambient=0.4, diffuse=0.6, specular=0.1, roughness=0.6),
            name="Blood Vessels", hoverinfo='text', text="<b>Vascular network</b>",
        ))
    return traces


def render_lymphatic_spread(organs: dict[str, Organ]) -> list:
    traces = [_torso_outline()]
    for name, organ in organs.items():
        if name == "Lymph Nodes":
            continue
        base = "#667788"
        pv_color = _per_vertex_color(base, organ.plotly_mesh['vertices'],
                                     noise_amp=6, vary_lighting=True)
        t = _organ_to_mesh3d(organ, per_vertex_colour=pv_color, opacity=0.40)
        traces.append(t)
    lymph = organs.get("Lymph Nodes")
    if lymph:
        base = REALISTIC_ORGAN_COLORS.get("Lymph Nodes", "#ffaa44")
        pv_color = _per_vertex_color(base, lymph.plotly_mesh['vertices'],
                                     noise_amp=8, vary_lighting=True)
        t = _organ_to_mesh3d(lymph, per_vertex_colour=pv_color, opacity=0.85)
        traces.append(t)
        if lymph.spread_regions:
            src = _organ_centre(lymph)
            for t_name in lymph.spread_regions:
                if t_name in organs:
                    tgt = _organ_centre(organs[t_name])
                    traces.extend(spread_path(src, tgt, "#ffaa44", n_particles=20))
    return traces


def render_tumour_localisation(organs: dict[str, Organ]) -> list:
    traces = [_torso_outline()]
    for name, organ in organs.items():
        has_tumour = organ.tumour_size > 0.01 or len(organ.lesions) > 0
        if has_tumour:
            base = REALISTIC_ORGAN_COLORS.get(name, organ.colour)
            pv_color = _per_vertex_color(base, organ.plotly_mesh['vertices'],
                                         noise_amp=10, vary_lighting=True)
        else:
            pv_color = _per_vertex_color("#334455", organ.plotly_mesh['vertices'],
                                         noise_amp=4, vary_lighting=False)
        t = _organ_to_mesh3d(organ, per_vertex_colour=pv_color,
                             opacity=0.85 if has_tumour else 0.20)
        traces.append(t)
        for les in organ.lesions:
            traces.extend(create_lesion_mesh(les.position, les.size * 1.3, min(1.0, les.opacity + 0.15),
                                             "#ff2222", glow=True, layers=les.layers, seed=les.seed))
        if organ.tumour_size > 0.1:
            centre = _organ_centre(organ)
            traces.extend(_bounding_box(centre, organ.tumour_size * 0.5, organ.colour))
    traces.extend(_pancreatic_tumor_traces(organs))
    return traces


def render_treatment_simulation(organs: dict[str, Organ], treatment_name: str = "",
                                response_prob: float = 0.0) -> list:
    traces = [_torso_outline()]
    for name, organ in organs.items():
        simulated_health = min(100, organ.health_score + response_prob * 30)
        simulated_risk = max(0, organ.risk_score - response_prob * 25)
        base = REALISTIC_ORGAN_COLORS.get(name, _health_to_colour(simulated_health))
        pv_color = _per_vertex_color(base, organ.plotly_mesh['vertices'],
                                     noise_amp=8, vary_lighting=True)
        t = _organ_to_mesh3d(organ, per_vertex_colour=pv_color, opacity=0.75)
        t.text = (f"<b>{name}</b><br>Projected Health: {simulated_health:.0f}%<br>"
                  f"Projected Risk: {simulated_risk:.0f}%<br>Treatment: {treatment_name or 'None'}")
        traces.append(t)
    return traces


def _torso_outline() -> go.Surface:
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(-1, 1, 60)
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
        opacity=0.04,
        colorscale=[[0, '#60a5fa'], [1, '#60a5fa']],
        showscale=False,
        contours=dict(
            x=dict(show=True, color='rgba(96,165,250,0.06)', width=1),
            y=dict(show=True, color='rgba(96,165,250,0.06)', width=1),
            z=dict(show=False),
        ),
        hoverinfo='skip',
    )


def _organ_centre(organ: Organ) -> tuple:
    md = organ.plotly_mesh
    if md is None:
        md = load_organ_mesh(organ.name)
        organ.plotly_mesh = md
    verts = md['vertices']
    if len(verts) == 0:
        return (0, 0, 0)
    return tuple(np.mean(verts, axis=0))


def _load_vessel_mesh():
    return load_organ_mesh("blood_vessels")


def _bounding_box(centre: tuple, half_size: float, colour: str) -> list:
    cx, cy, cz = centre
    hs = half_size
    corners = np.array([
        [cx - hs, cy - hs, cz - hs], [cx + hs, cy - hs, cz - hs],
        [cx + hs, cy + hs, cz - hs], [cx - hs, cy + hs, cz - hs],
        [cx - hs, cy - hs, cz + hs], [cx + hs, cy - hs, cz + hs],
        [cx + hs, cy + hs, cz + hs], [cx - hs, cy + hs, cz + hs],
    ])
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    traces = []
    for i, j in edges:
        traces.append(go.Scatter3d(
            x=[corners[i, 0], corners[j, 0]],
            y=[corners[i, 1], corners[j, 1]],
            z=[corners[i, 2], corners[j, 2]],
            mode='lines', line=dict(color=colour, width=2, dash='dash'),
            hoverinfo='skip', showlegend=False,
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


def render_compartmental_twin(organs: dict, forecast_data: dict | None = None):
    traces = [_torso_outline()]
    for name, organ in organs.items():
        base = REALISTIC_ORGAN_COLORS.get(name, organ.colour)
        pv_color = _per_vertex_color(base, organ.plotly_mesh['vertices'],
                                     noise_amp=8, vary_lighting=True)
        mesh = _organ_to_mesh3d(organ, per_vertex_colour=pv_color, opacity=organ.opacity)
        if mesh:
            traces.append(mesh)
        for lesion in organ.lesions:
            lmesh = create_lesion_mesh(
                lesion.position, lesion.size * 1.2,
                lesion.opacity, "#e74c3c",
                glow=True, n_points=60,
                layers=lesion.layers, seed=lesion.seed,
            )
            traces.extend(lmesh)
            if organ.risk_score >= 50:
                pulse = organ_pulse(
                    ORGAN_CENTRES.get(name, (0, 0, 0)),
                    organ.risk_score, organ.colour,
                )
                traces.extend(pulse)
        if organ.risk_score > 40 and hasattr(organ, 'spread_regions'):
            for target in getattr(organ, 'spread_regions', []):
                tgt_centre = ORGAN_CENTRES.get(target)
                src_centre = ORGAN_CENTRES.get(name)
                if src_centre and tgt_centre:
                    traces.extend(spread_path(src_centre, tgt_centre, organ.colour,
                                              n_particles=8, line_width=1.5))
    if forecast_data:
        risk = float(forecast_data['risk'][-1])
        tumor = float(forecast_data['T'][-1])
        ca = float(forecast_data['CA'][-1])
        health = float(forecast_data['health'][-1])
        info = (f"Compartmental Model @ day {len(forecast_data['days'])-1}"
                f"<br>Risk: {risk:.1f}% | Health: {health:.1f}%"
                f"<br>Tumor: {tumor:.2f} cm³ | CA19-9: {ca:.0f} U/mL")
        traces.append(go.Scatter3d(
            x=[-1.8], y=[0.3], z=[2.4],
            mode='text', text=[f'<b>Compartmental Twin</b>'],
            textfont=dict(size=11, color='#2563eb', family='Inter, sans-serif'),
            hoverinfo='text', hovertext=[info], showlegend=False,
        ))
    traces.extend(_pancreatic_tumor_traces(organs))
    return traces


RENDER_MODES = {
    "Normal Anatomy": render_normal,
    "Risk Heatmap": render_risk_heatmap,
    "Disease Progression": render_disease_progression,
    "Blood Supply": render_blood_supply,
    "Lymphatic Spread": render_lymphatic_spread,
    "Tumour Localisation": render_tumour_localisation,
    "Treatment Response": render_treatment_simulation,
    "Compartmental Twin": render_compartmental_twin,
    "Realistic Anatomy": lambda organs: _realistic_anatomy_traces(organs),
}


def _realistic_anatomy_traces(organs) -> list:
    from .real_ct_twin import render_realistic_twin
    return render_realistic_twin(organs)