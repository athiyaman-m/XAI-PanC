import numpy as np
import plotly.graph_objects as go
from typing import Optional


def _radial_function(u: np.ndarray, v: np.ndarray, rng) -> np.ndarray:
    """Lobulated, irregular radial multiplier for a tumour-region surface.

    Built from a few seeded spherical-harmonic-like modes plus a fine
    lobulation term, so the resulting region looks like a real invasive mass
    rather than a perfect sphere.
    """
    return np.maximum(
        1.0
        + 0.16 * np.sin(u * 2 + rng.uniform(0, 2 * np.pi)) * np.cos(v * 3 + rng.uniform(0, 2 * np.pi))
        + 0.10 * np.sin(u * 3 + rng.uniform(0, 2 * np.pi)) * np.sin(v * 2 + rng.uniform(0, 2 * np.pi))
        + 0.06 * np.cos(u * 4 + rng.uniform(0, 2 * np.pi)) * np.cos(v * 4 + rng.uniform(0, 2 * np.pi))
        + 0.05 * np.sin(u * 5 + v * 3 + rng.uniform(0, 2 * np.pi)),
        0.30,
    )


def build_lesion_surface(position, size=0.08, n_u=36, n_v=24, seed=7,
                         layer=1.0) -> tuple[np.ndarray, np.ndarray]:
    """Build an explicit, closed, lobulated surface mesh of a tumour region.

    The region is sampled on a spherical (n_u x n_v) grid whose radius is a
    seeded radial function, then triangulated into a real surface mesh with
    explicit ``i/j/k`` faces (no ``alphahull`` point clouds). ``layer`` scales
    the radius so the same builder can produce concentric surface shells.
    """
    cx, cy, cz = position
    rng = np.random.RandomState(seed)
    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    v = np.linspace(0, np.pi, n_v)
    U, V = np.meshgrid(u, v, indexing='ij')
    r = max(size, 1e-4) * max(layer, 1e-3) * _radial_function(U, V, rng)
    X = cx + r * np.sin(V) * np.cos(U)
    Y = cy + r * np.sin(V) * np.sin(U)
    Z = cz + r * np.cos(V)
    verts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    faces = []
    for a in range(n_u - 1):
        for b in range(n_v - 1):
            p1 = a * n_v + b
            p2 = (a + 1) * n_v + b
            p3 = (a + 1) * n_v + (b + 1)
            p4 = a * n_v + (b + 1)
            faces.append([p1, p2, p3])
            faces.append([p1, p3, p4])
    return verts, np.array(faces, dtype=np.int32)


def create_lesion_mesh(position, size=0.08, opacity=0.7, colour="#ff3333",
                       glow=True, n_points=300, layers=3, seed=7):
    """Render a tumour/cancer region as a *surface-layered* 3D region.

    Instead of scattering single coordinate points, the region is drawn as a
    proper closed surface mesh (lobulated, seeded) surrounded by concentric
    translucent surface shells (the "layers"), plus an optional outer glow
    surface — so the mass reads as a layered volume on the 3D model.

    ``layers`` controls how many concentric surface shells are drawn; ``seed``
    fixes the region's shape so the same lesion keeps its outline between
    frames (only ``size`` changes on growth).
    """
    cx, cy, cz = position
    r_comp = int(colour[1:3], 16)
    g_comp = int(colour[3:5], 16)
    b_comp = int(colour[5:7], 16)

    n_layers = max(1, int(layers))
    n_u = max(12, (int(n_points) // 12) * 2)
    n_v = max(8, int(n_points) // 24)

    traces = []
    # Outermost shell first (behind), then progressively smaller/inner shells on
    # top at lower opacity, giving the onion-layered tumour-region look.
    shell_fractions = np.linspace(1.0, 0.45, n_layers)
    for i, frac in enumerate(shell_fractions):
        verts, faces = build_lesion_surface(position, size, n_u=n_u, n_v=n_v,
                                            seed=seed, layer=frac)
        if len(verts) == 0 or len(faces) == 0:
            continue
        # inner shells slightly darker (deeper tissue) and more translucent
        deep = 1.0 - 0.28 * (i / max(n_layers - 1, 1))
        layer_op = opacity if i == 0 else opacity * (0.32 + 0.28 * (i / max(n_layers - 1, 1)))
        rng = np.random.RandomState(seed + i)
        noise = rng.randn(len(verts)) * 12
        vertex_colors = np.column_stack([
            np.clip(r_comp * deep + noise, 0, 255),
            np.clip(g_comp * deep + noise * 0.6, 0, 255),
            np.clip(b_comp * deep + noise * 0.3, 0, 255),
        ]) / 255.0
        traces.append(go.Mesh3d(
            x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
            opacity=layer_op, vertexcolor=vertex_colors,
            showscale=False,
            name="Lesion Region" if i == 0 else f"Lesion Layer {i + 1}",
            lighting=dict(ambient=0.35, diffuse=0.75, specular=0.3, roughness=0.4),
            hoverinfo="text",
            text=(f"<b>Lesion region</b><br>Size: {size*100:.1f} mm<br>"
                  f"Surface layers: {n_layers}<br>Confidence: {opacity*100:.0f}%"
                  if i == 0 else None),
        ))

    if glow:
        g_verts, g_faces = build_lesion_surface(
            position, size * 1.6,
            n_u=max(12, n_u // 2), n_v=max(8, n_v // 2),
            seed=seed + 100, layer=1.0,
        )
        if len(g_verts) > 0 and len(g_faces) > 0:
            traces.append(go.Mesh3d(
                x=g_verts[:, 0], y=g_verts[:, 1], z=g_verts[:, 2],
                i=g_faces[:, 0], j=g_faces[:, 1], k=g_faces[:, 2],
                opacity=opacity * 0.18,
                color=colour, showscale=False,
                name="Lesion Glow",
                lighting=dict(ambient=0.6, diffuse=0.3, specular=0.05, roughness=0.9),
                hoverinfo="skip",
            ))
    return traces


def create_metastasis_chain(positions, colour="#ff6666", opacity=0.5):
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
