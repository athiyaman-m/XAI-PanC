import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import io
from datetime import datetime, timedelta

from prediction_engine import run_full_prediction
from digital_twin import DigitalTwinEngine, RENDER_MODES, organ_hover_html, patient_panel_html

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ExAI-PanC",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CSS — Professional medical-dark theme
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.stApp{background:linear-gradient(135deg,#060b14 0%,#0b1222 50%,#080e1a 100%);color:#dce3ed;}

/* Hero */
.hero{background:linear-gradient(120deg,#0a1a2e 0%,#0d2640 45%,#14304d 100%);
      border:1px solid #1a4a7a;border-radius:16px;padding:1.8rem 2.5rem;margin-bottom:1.5rem;position:relative;overflow:hidden;}
.hero::after{content:'';position:absolute;top:-35%;right:-5%;width:340px;height:340px;
             background:radial-gradient(circle,rgba(59,130,246,0.1) 0%,transparent 70%);pointer-events:none;}
.hero-badge{display:inline-block;background:rgba(59,130,246,.12);border:1px solid rgba(59,130,246,.3);
            color:#60a5fa;font-size:.65rem;font-weight:600;letter-spacing:1.2px;text-transform:uppercase;
            padding:3px 10px;border-radius:20px;margin-bottom:.6rem;}
.hero-title{font-size:2.1rem;font-weight:700;color:#fff;margin:0 0 .2rem;letter-spacing:-.5px;}
.hero-title span{color:#60a5fa;}
.hero-sub{font-size:.85rem;color:#7a9fc5;margin:0;}

/* Cards */
.card{background:linear-gradient(145deg,#0e1e30,#0b1828);border:1px solid #1a3555;
      border-radius:10px;padding:1.1rem 1.3rem;text-align:center;transition:border-color .2s;}
.card:hover{border-color:#3b82f6;}
.cv{font-size:1.7rem;font-weight:700;line-height:1.1;margin-bottom:.2rem;}
.cl{font-size:.7rem;color:#7a9fc5;text-transform:uppercase;letter-spacing:.8px;font-weight:500;}
.cs{font-size:.65rem;color:#3a5a7a;margin-top:.15rem;}

/* Result banners */
.res-hi{background:linear-gradient(120deg,#280808,#350d0d);border:1.5px solid #b03020;
        border-radius:13px;padding:1.3rem 1.6rem;margin:1rem 0;}
.res-lo{background:linear-gradient(120deg,#082515,#0a301a);border:1.5px solid #229955;
        border-radius:13px;padding:1.3rem 1.6rem;margin:1rem 0;}
.res-title{font-size:1.25rem;font-weight:700;margin-bottom:.3rem;}

/* Info/warn */
.info-box{background:rgba(59,130,246,.06);border:1px solid rgba(59,130,246,.2);
          border-radius:8px;padding:.8rem 1.1rem;font-size:.82rem;color:#8bb8e0;line-height:1.6;}
.warn-box{background:rgba(245,158,11,.07);border:1px solid rgba(245,158,11,.25);
          border-radius:8px;padding:.8rem 1.1rem;font-size:.8rem;color:#e8b84a;line-height:1.6;}

/* Row item */
.bm-row{display:flex;justify-content:space-between;align-items:center;
        padding:.45rem .7rem;margin-bottom:.3rem;
        background:#0e1e30;border-radius:7px;border:1px solid #1a3555;}
.mono{font-family:'JetBrains Mono',monospace;font-size:.8rem;color:#60a5fa;}

/* Section header */
.sec-hdr{font-size:.85rem;font-weight:600;color:#60a5fa;border-left:3px solid #60a5fa;
         padding-left:10px;margin:1.3rem 0 .7rem;text-transform:uppercase;letter-spacing:.5px;}

/* Sidebar */
[data-testid="stSidebar"]{background:#050a12!important;border-right:1px solid #1a3555;}
[data-testid="stSidebar"] label{color:#7a9fc5!important;font-size:.8rem!important;}
[data-testid="stSidebar"] .stSelectbox label, [data-testid="stSidebar"] .stSlider label{font-size:.8rem!important;}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{background:transparent;border-bottom:1px solid #1a3555;gap:2px;}
.stTabs [data-baseweb="tab"]{background:transparent;color:#4a6a8a;border-radius:8px 8px 0 0;
                               padding:6px 16px;font-size:.82rem;font-weight:500;}
.stTabs [aria-selected="true"]{background:rgba(59,130,246,.1)!important;
                                 color:#60a5fa!important;border-bottom:2px solid #60a5fa;}

/* Buttons */
.stButton>button{background:linear-gradient(135deg,#1560a0,#1e80c8);color:#fff;
                  border:none;border-radius:8px;padding:.55rem 1.5rem;
                  font-weight:600;font-size:.88rem;width:100%;transition:all .2s;}
.stButton>button:hover{background:linear-gradient(135deg,#1e80c8,#3498db);
                        transform:translateY(-1px);box-shadow:0 4px 18px rgba(52,152,219,.3);}

/* Organ risk pills */
.organ-pill{display:inline-block;padding:3px 10px;border-radius:20px;
            font-size:.72rem;font-weight:600;margin:3px;letter-spacing:.5px;}

/* Footer */
.footer{text-align:center;color:#1e3a5a;font-size:.7rem;
        margin-top:2.5rem;padding:1rem;border-top:1px solid #1a3555;}

/* Upload zone */
.upload-zone{background:rgba(59,130,246,.04);border:1.5px dashed rgba(59,130,246,.25);
             border-radius:10px;padding:.8rem;margin-bottom:.6rem;text-align:center;}

/* Sidebar section titles */
.side-sec{font-size:.7rem;font-weight:600;color:#60a5fa;text-transform:uppercase;
          letter-spacing:.8px;margin:.8rem 0 .4rem;padding-bottom:.3rem;border-bottom:1px solid #1a3555;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PREDICTION ENGINE
# ─────────────────────────────────────────────────────────────
# All predictions via prediction_engine.py


# ─────────────────────────────────────────────────────────────
# AI TWIN — SVG BODY MAP
# ─────────────────────────────────────────────────────────────

def render_ai_twin(organ_risks):
    def rgba(hex_col, alpha=0.75):
        h = hex_col.lstrip("#")
        r,g,b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
        return f"rgba({r},{g},{b},{alpha})"

    pr    = organ_risks["Pancreas"]
    lr    = organ_risks["Liver"]
    br    = organ_risks["Bile Duct"]
    dr    = organ_risks["Duodenum"]
    sr_   = organ_risks["Stomach"]
    spl   = organ_risks["Spleen"]
    lnr   = organ_risks["Lymph Nodes"]
    lunr  = organ_risks["Lung"]

    svg = f"""
<svg viewBox="0 0 320 580" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:340px;display:block;margin:auto;filter:drop-shadow(0 0 20px rgba(30,100,180,0.15));">
  <defs>
    <radialGradient id="glow_panc" cx="50%" cy="50%" r="50%">
      <stop offset="0%" stop-color="{pr[1]}" stop-opacity="0.9"/>
      <stop offset="100%" stop-color="{pr[1]}" stop-opacity="0.2"/>
    </radialGradient>
    <filter id="blur"><feGaussianBlur stdDeviation="2"/></filter>
    <filter id="glow_f">
      <feGaussianBlur stdDeviation="3" result="blur"/>
      <feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>

  <!-- Body outline -->
  <g opacity="0.15" fill="#3b82f6">
    <ellipse cx="160" cy="52" rx="38" ry="44"/>
    <rect x="148" y="88" width="24" height="22" rx="8"/>
    <path d="M105,108 Q80,115 75,180 Q72,260 78,310 Q85,340 100,350
             L130,360 L160,365 L190,360 L220,350
             Q235,340 242,310 Q248,260 245,180 Q240,115 215,108 Z" />
    <path d="M105,115 Q75,130 65,200 Q58,240 62,280 Q65,300 72,305
             Q80,308 84,295 Q88,260 92,220 Q98,175 108,145 Z"/>
    <path d="M215,115 Q245,130 255,200 Q262,240 258,280 Q255,300 248,305
             Q240,308 236,295 Q232,260 228,220 Q222,175 212,145 Z"/>
    <path d="M120,358 Q105,380 100,440 Q96,490 98,530 Q100,550 112,552
             Q124,554 128,535 Q132,490 135,440 Q138,400 140,365 Z"/>
    <path d="M200,358 Q215,380 220,440 Q224,490 222,530 Q220,550 208,552
             Q196,554 192,535 Q188,490 185,440 Q182,400 180,365 Z"/>
  </g>

  <g fill="none" stroke="#3b82f6" stroke-width="1.3" opacity="0.4">
    <ellipse cx="160" cy="52" rx="38" ry="44"/>
    <path d="M148,88 L148,108 M172,88 L172,108"/>
    <path d="M105,108 Q80,115 75,180 Q72,260 78,310 Q85,340 100,350
             L130,360 L160,365 L190,360 L220,350
             Q235,340 242,310 Q248,260 245,180 Q240,115 215,108 Z"/>
    <path d="M105,115 Q75,130 65,200 Q58,240 62,280 Q65,300 72,305"/>
    <path d="M215,115 Q245,130 255,200 Q262,240 258,280 Q255,300 248,305"/>
    <path d="M120,358 Q105,380 100,440 Q96,490 98,530 Q100,550 112,552"/>
    <path d="M200,358 Q215,380 220,440 Q224,490 222,530 Q220,550 208,552"/>
    <path d="M148,112 Q130,116 115,122 M172,112 Q190,116 205,122"/>
    <ellipse cx="160" cy="185" rx="58" ry="52" stroke-dasharray="4,3" opacity="0.5"/>
    <line x1="160" y1="108" x2="160" y2="360" stroke-dasharray="5,4" opacity="0.3"/>
    <path d="M105,340 Q130,355 160,358 Q190,355 215,340"/>
  </g>

  <!-- Lungs -->
  <g filter="url(#glow_f)">
    <ellipse cx="131" cy="168" rx="30" ry="42" fill="{rgba(lunr[1],0.6)}" stroke="{lunr[1]}" stroke-width="1.5"/>
    <ellipse cx="189" cy="168" rx="30" ry="42" fill="{rgba(lunr[1],0.6)}" stroke="{lunr[1]}" stroke-width="1.5"/>
    <text x="131" y="165" text-anchor="middle" fill="#fff" font-size="8" font-weight="600">L</text>
    <text x="189" y="165" text-anchor="middle" fill="#fff" font-size="8" font-weight="600">R</text>
    <text x="160" y="178" text-anchor="middle" fill="#fff" font-size="7" opacity="0.8">Lung</text>
    <text x="160" y="187" text-anchor="middle" fill="#fff" font-size="8" font-weight="700">{lunr[0]}%</text>
  </g>

  <!-- Heart -->
  <g>
    <path d="M152,148 Q152,140 160,148 Q168,140 168,148 Q168,158 160,166 Q152,158 152,148Z"
          fill="#e74c3c" opacity="0.7" stroke="#ff6b6b" stroke-width="1"/>
  </g>

  <!-- Liver -->
  <g filter="url(#glow_f)">
    <path d="M168,218 Q200,215 210,228 Q218,242 210,258 Q200,268 185,270 Q172,270 168,258 Z"
          fill="{rgba(lr[1],0.72)}" stroke="{lr[1]}" stroke-width="1.8"/>
    <text x="190" y="244" text-anchor="middle" fill="#fff" font-size="8" font-weight="600">Liver</text>
    <text x="190" y="255" text-anchor="middle" fill="#fff" font-size="9" font-weight="700">{lr[0]}%</text>
  </g>

  <!-- Spleen -->
  <g filter="url(#glow_f)">
    <ellipse cx="118" cy="240" rx="22" ry="18"
             fill="{rgba(spl[1],0.68)}" stroke="{spl[1]}" stroke-width="1.5"/>
    <text x="118" y="238" text-anchor="middle" fill="#fff" font-size="7" font-weight="600">Spleen</text>
    <text x="118" y="248" text-anchor="middle" fill="#fff" font-size="8" font-weight="700">{spl[0]}%</text>
  </g>

  <!-- Stomach -->
  <g filter="url(#glow_f)">
    <path d="M138,218 Q158,212 168,224 Q172,236 165,248 Q155,256 140,252 Q128,244 128,232 Q130,220 138,218Z"
          fill="{rgba(sr_[1],0.65)}" stroke="{sr_[1]}" stroke-width="1.5"/>
    <text x="148" y="234" text-anchor="middle" fill="#fff" font-size="7" font-weight="600">Stomach</text>
    <text x="148" y="244" text-anchor="middle" fill="#fff" font-size="8" font-weight="700">{sr_[0]}%</text>
  </g>

  <!-- Pancreas -->
  <g filter="url(#glow_f)">
    <ellipse cx="160" cy="272" rx="38" ry="14"
             fill="url(#glow_panc)" stroke="{pr[1]}" stroke-width="2.5"/>
    <text x="160" y="269" text-anchor="middle" fill="#fff" font-size="8" font-weight="700">PANCREAS</text>
    <text x="160" y="280" text-anchor="middle" fill="#fff" font-size="9" font-weight="800">{pr[0]}%</text>
    <ellipse cx="160" cy="272" rx="46" ry="20"
             fill="none" stroke="{pr[1]}" stroke-width="1" opacity="0.4" stroke-dasharray="6,4"/>
  </g>

  <!-- Bile Duct -->
  <g filter="url(#glow_f)">
    <path d="M176,260 Q180,268 178,280 Q176,290 172,295"
          fill="none" stroke="{br[1]}" stroke-width="5" stroke-linecap="round" opacity="0.8"/>
    <circle cx="172" cy="298" r="8" fill="{rgba(br[1],0.7)}" stroke="{br[1]}" stroke-width="1.5"/>
    <text x="172" y="302" text-anchor="middle" fill="#fff" font-size="6" font-weight="600">Bile</text>
    <text x="188" y="282" text-anchor="start" fill="{br[1]}" font-size="7" font-weight="700">{br[0]}%</text>
  </g>

  <!-- Duodenum -->
  <g filter="url(#glow_f)">
    <path d="M178,286 Q195,290 198,304 Q198,318 188,320 Q178,318 176,308"
          fill="none" stroke="{dr[1]}" stroke-width="5" stroke-linecap="round" opacity="0.75"/>
    <text x="198" y="310" text-anchor="start" fill="{dr[1]}" font-size="7" font-weight="700">{dr[0]}%</text>
    <text x="188" y="320" text-anchor="middle" fill="{dr[1]}" font-size="6">Duod.</text>
  </g>

  <!-- Lymph Nodes -->
  <g opacity="0.85">
    <circle cx="108" cy="200" r="5" fill="{rgba(lnr[1],0.8)}" stroke="{lnr[1]}" stroke-width="1"/>
    <circle cx="212" cy="200" r="5" fill="{rgba(lnr[1],0.8)}" stroke="{lnr[1]}" stroke-width="1"/>
    <circle cx="140" cy="295" r="5" fill="{rgba(lnr[1],0.8)}" stroke="{lnr[1]}" stroke-width="1"/>
    <circle cx="180" cy="295" r="5" fill="{rgba(lnr[1],0.8)}" stroke="{lnr[1]}" stroke-width="1"/>
    <text x="160" y="312" text-anchor="middle" fill="{lnr[1]}" font-size="7">Lymph {lnr[0]}%</text>
  </g>

  <!-- Risk legend -->
  <g transform="translate(60, 420)">
    <text x="0" y="0" fill="#7a9fc5" font-size="9" font-weight="600">RISK SCALE</text>
    <rect x="0"   y="8" width="14" height="8" fill="#2ecc71" rx="2"/>
    <text x="17"  y="17" fill="#8bb8e0" font-size="8">Low &lt;20%</text>
    <rect x="0"   y="22" width="14" height="8" fill="#f1c40f" rx="2"/>
    <text x="17"  y="31" fill="#8bb8e0" font-size="8">Mild 20-44%</text>
    <rect x="0"   y="36" width="14" height="8" fill="#f39c12" rx="2"/>
    <text x="17"  y="45" fill="#8bb8e0" font-size="8">Moderate 45-69%</text>
    <rect x="0"   y="50" width="14" height="8" fill="#e74c3c" rx="2"/>
    <text x="17"  y="59" fill="#8bb8e0" font-size="8">High ≥70%</text>
  </g>

  <text x="160" y="410" text-anchor="middle" fill="#60a5fa" font-size="10" font-weight="700"
        letter-spacing="2">AI DIGITAL TWIN</text>
  <text x="160" y="422" text-anchor="middle" fill="#3a5a7a" font-size="8">Organ risk simulation</text>
</svg>"""
    return svg


# ─────────────────────────────────────────────────────────────
# 3D DIGITAL TWIN — Realistic full-torso organ model
# ─────────────────────────────────────────────────────────────

def render_3d_digital_twin(organ_risks):
    """
    Creates a photorealistic interactive 3D digital twin of the human torso
    with all major organs positioned anatomically. Each organ is color-coded
    by risk level using Mesh3d alpha shapes for smooth organic surfaces.

    Risk colours:  green (<20%) -> yellow (20-44%) -> orange (45-69%) -> red (>=70%)
    Affected organs glow and show lesion hotspots when risk exceeds thresholds.
    """
    np.random.seed(42)

    # ── Risk colour helpers ──
    def rc(pct, a=0.88):
        if pct >= 70: return f'rgba(220,50,50,{a})'
        if pct >= 45: return f'rgba(243,156,18,{a})'
        if pct >= 20: return f'rgba(241,196,15,{a})'
        return f'rgba(46,204,113,{a})'

    def rhex(pct):
        if pct >= 70: return '#dc3232'
        if pct >= 45: return '#f39c12'
        if pct >= 20: return '#f1c40f'
        return '#2ecc71'

    def r_bright(pct):
        if pct >= 70: return '#ff3333'
        if pct >= 45: return '#ffb347'
        if pct >= 20: return '#ffe066'
        return '#5ced73'

    # ── Mesh helper (alpha-shape from point cloud) ──
    def mesh(pts, risk_val, opacity=0.87, name='', color=None):
        col = color or rc(risk_val, opacity)
        pts = np.array(pts)
        return go.Mesh3d(
            x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
            alphahull=5, opacity=opacity,
            color=col, name=name,
            showscale=False,
            lighting=dict(ambient=0.40, diffuse=0.78, specular=0.25, roughness=0.35),
            hoverinfo='text',
            text=f'<b>{name}</b><br>Risk: {risk_val}%<br>Status: {"HIGH" if risk_val>=70 else "MODERATE" if risk_val>=45 else "MILD" if risk_val>=20 else "LOW"}',
        )

    # ── Point cloud generators ──
    def ellipsoid_pts(cx, cy, cz, rx, ry, rz, n=500):
        pts = []
        for _ in range(n):
            t = np.random.uniform(0, 2*np.pi)
            p = np.random.uniform(0, np.pi)
            r = np.random.uniform(0, 1)**(1/3)
            pts.append([cx + r*rx*np.sin(p)*np.cos(t),
                        cy + r*ry*np.sin(p)*np.sin(t),
                        cz + r*rz*np.cos(p)])
        return pts

    def tube_pts(path, radii, n_circ=16):
        path = np.array(path)
        n = len(path)
        pts = []
        for i in range(n):
            r = radii[i] if hasattr(radii, '__len__') else radii
            if i == 0: d = path[1] - path[0]
            elif i == n-1: d = path[-1] - path[-2]
            else: d = path[i+1] - path[i-1]
            d_norm = np.linalg.norm(d) + 1e-10
            d = d / d_norm
            if abs(d[0]) < 0.9:
                p1 = np.cross(d, [1, 0, 0])
            else:
                p1 = np.cross(d, [0, 1, 0])
            p1 = p1 / (np.linalg.norm(p1) + 1e-10)
            p2 = np.cross(d, p1)
            for j in range(n_circ):
                a = 2*np.pi * j / n_circ
                pts.append(path[i] + r * (np.cos(a)*p1 + np.sin(a)*p2))
        return pts

    def deform_ellipsoid(cx, cy, cz, rx, ry, rz, deform, n=600):
        pts = []
        for _ in range(n):
            t = np.random.uniform(0, 2*np.pi)
            p = np.random.uniform(0, np.pi)
            r = np.random.uniform(0, 1)**(1/3)
            x = r*rx*np.sin(p)*np.cos(t)
            y = r*ry*np.sin(p)*np.sin(t)
            z = r*rz*np.cos(p)
            dx, dy, dz = deform(x, y, z)
            pts.append([cx+x+dx, cy+y+dy, cz+z+dz])
        return pts

    # ── Extract risks ──
    panc_r = organ_risks["Pancreas"][0]
    liver_r = organ_risks["Liver"][0]
    bile_r = organ_risks["Bile Duct"][0]
    duo_r = organ_risks["Duodenum"][0]
    stom_r = organ_risks["Stomach"][0]
    spl_r = organ_risks["Spleen"][0]
    lymph_r = organ_risks["Lymph Nodes"][0]
    lung_r = organ_risks["Lung"][0]

    # ═══════════════════════════════════════════════════════
    # 1. TORSO — semi-transparent body outline
    # ═══════════════════════════════════════════════════════
    u = np.linspace(0, 2*np.pi, 80)
    v = np.linspace(-1, 1, 100)
    U, V = np.meshgrid(u, v)

    w = 1.5 * (1 - 0.12*V**2 + 0.04*V**4)
    d = 0.75 * (1 - 0.10*V**2 + 0.02*V**4)

    X_t = w * np.cos(U)
    Y_t = d * np.sin(U)
    Z_t = 2.2 * V

    shoulder_mask = Z_t > 1.6
    X_t[shoulder_mask] *= 1 + 0.15*(Z_t[shoulder_mask]-1.6)/0.6

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X_t, y=Y_t, z=Z_t,
        opacity=0.04,
        colorscale=[[0, '#3b82f6'], [1, '#3b82f6']],
        showscale=False,
        contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
        hoverinfo='skip',
    ))

    # Body wireframe outline
    fig.add_trace(go.Surface(
        x=X_t, y=Y_t, z=Z_t,
        opacity=0.0, showscale=False,
        contours=dict(
            x=dict(show=True, color='rgba(59,130,246,0.08)', width=1),
            y=dict(show=True, color='rgba(59,130,246,0.08)', width=1),
            z=dict(show=False),
        ),
        hoverinfo='skip',
    ))

    # ═══════════════════════════════════════════════════════
    # 2. LUNGS — right (larger) and left (with cardiac notch)
    # ═══════════════════════════════════════════════════════
    lung_risk = lung_r

    def lung_deform(x, y, z, side=1):
        dx = 0
        dy = -0.15 * (z/1.2)**2  # concave mediastinal surface
        dz = 0
        if side < 0 and y > 0 and z < 0.1:
            dy += 0.1              # cardiac notch on left lung
        return dx, dy, dz

    for side, label in [(-1, 'Left Lung'), (1, 'Right Lung')]:
        lung_col = rc(lung_risk, 0.75)
        pts = deform_ellipsoid(
            side*0.65, 0.05, 1.2,
            0.35, 0.22, 0.55,
            lambda x, y, z: lung_deform(x, y, z, side),
            n=700
        )
        fig.add_trace(mesh(pts, lung_risk, 0.75, label, lung_col))

    # ═══════════════════════════════════════════════════════
    # 3. HEART
    # ═══════════════════════════════════════════════════════
    heart_risk = max(0, int(panc_r * 0.35))
    heart_col = rc(heart_risk, 0.7)
    heart_pts = deform_ellipsoid(-0.1, 0.25, 1.0, 0.25, 0.18, 0.30,
        lambda x, y, z: (0.08*x**2, 0.05*z, -0.05*y**2), n=400)
    fig.add_trace(mesh(heart_pts, heart_risk, 0.70, 'Heart', heart_col))

    # ═══════════════════════════════════════════════════════
    # 4. LIVER — large right lobe + smaller left lobe
    # ═══════════════════════════════════════════════════════
    def liver_deform(x, y, z):
        # Right lobe larger, left lobe extends left
        if x > 0:
            return (0.1*z, 0.05*y, -0.02*x**2)
        else:
            return (0.1*z-0.2, 0.05*y+0.08, -0.05*x)

    liver_pts = deform_ellipsoid(0.55, 0.2, 0.55, 0.55, 0.25, 0.35,
        liver_deform, n=900)
    fig.add_trace(mesh(liver_pts, liver_r, 0.80, 'Liver'))

    # ═══════════════════════════════════════════════════════
    # 5. SPLEEN — bean-shaped
    # ═══════════════════════════════════════════════════════
    def spleen_deform(x, y, z):
        return (0.15*z, -0.1*x + 0.05*np.sin(z*4), 0)

    spleen_pts = deform_ellipsoid(-1.05, -0.1, 0.35, 0.20, 0.12, 0.28,
        spleen_deform, n=500)
    fig.add_trace(mesh(spleen_pts, spl_r, 0.78, 'Spleen'))

    # ═══════════════════════════════════════════════════════
    # 6. STOMACH — J-shaped
    # ═══════════════════════════════════════════════════════
    stom_path = [
        [-0.2, 0.45, 0.5], [-0.25, 0.5, 0.4], [-0.3, 0.5, 0.3],
        [-0.35, 0.45, 0.2], [-0.4, 0.35, 0.1], [-0.35, 0.2, 0.0],
        [-0.25, 0.1, -0.1], [-0.15, 0.0, -0.2], [-0.05, -0.05, -0.25],
        [0.05, -0.05, -0.25], [0.15, 0.0, -0.2],
    ]
    stom_radii = np.linspace(0.25, 0.15, len(stom_path))
    stom_pts = tube_pts(stom_path, stom_radii, 18)
    fig.add_trace(mesh(stom_pts, stom_r, 0.72, 'Stomach'))

    # ═══════════════════════════════════════════════════════
    # 7. PANCREAS — detailed with head/body/tail regions
    # ═══════════════════════════════════════════════════════
    # Centreline follows anatomical path: head (right) -> body -> tail (left, up)
    panc_n = 100
    s = np.linspace(0, 1, panc_n)

    panc_cx = 0.8 - 1.8 * s + 0.2 * s**2
    panc_cy = 0.0 + 0.3 * s - 0.5 * s**2
    panc_cz = -0.1 + 0.5 * s

    # Radius profile: head is bulbous, body is uniform, tail tapers
    panc_r_vals = np.zeros(panc_n)
    for i, si in enumerate(s):
        if si < 0.18:
            panc_r_vals[i] = 0.22 * (1 + 0.25*np.sin(si/0.18*np.pi))
        elif si < 0.28:
            panc_r_vals[i] = 0.22 - (si-0.18)/0.10 * 0.12
        elif si < 0.75:
            panc_r_vals[i] = 0.10
        else:
            panc_r_vals[i] = 0.10 * (1 - (si-0.75)/0.25 * 0.55)

    panc_pts = tube_pts(list(zip(panc_cx, panc_cy, panc_cz)), panc_r_vals, 20)

    # Per-vertex risk mapping for pancreas
    panc_risk_map = np.zeros(len(panc_pts))
    pts_per_slice = 20
    for i in range(panc_n):
        si = s[i]
        if si < 0.22:
            seg_r = min(1.0, panc_r/100 * 1.20)
        elif si < 0.35:
            seg_r = panc_r/100
        elif si < 0.75:
            seg_r = panc_r/100 * 0.85
        else:
            seg_r = panc_r/100 * 0.55

        if panc_r/100 > 0.40:
            lesion = 0.35 * np.exp(-((si - 0.06)**2) / 0.006)
            seg_r = min(1.0, seg_r + lesion)

        for j in range(pts_per_slice):
            idx = i * pts_per_slice + j
            if idx < len(panc_pts):
                panc_risk_map[idx] = seg_r

    # Create gradient-coloured pancreas mesh using per-vertex colour
    panc_colors = []
    for rv in panc_risk_map:
        if rv >= 0.70: panc_colors.append('#dc3232')
        elif rv >= 0.45: panc_colors.append('#f39c12')
        elif rv >= 0.20: panc_colors.append('#f1c40f')
        else: panc_colors.append('#2ecc71')

    panc_pts_arr = np.array(panc_pts)
    fig.add_trace(go.Mesh3d(
        x=panc_pts_arr[:, 0], y=panc_pts_arr[:, 1], z=panc_pts_arr[:, 2],
        alphahull=5, opacity=0.92,
        vertexcolor=panc_colors,
        name='Pancreas',
        showscale=False,
        lighting=dict(ambient=0.40, diffuse=0.78, specular=0.25, roughness=0.35),
        hoverinfo='text',
        text=f'<b>Pancreas</b><br>Risk: {panc_r}%',
    ))

    # ── Pancreas region labels ──
    label_positions = [
        (0.08, 'HEAD', panc_r/100 * 1.2),
        (0.40, 'BODY', panc_r/100 * 0.85),
        (0.90, 'TAIL', panc_r/100 * 0.55),
    ]
    for si, lbl, lr in label_positions:
        idx = min(int(si * panc_n), panc_n-1)
        lcol = rhex(lr * 100)
        fig.add_trace(go.Scatter3d(
            x=[panc_cx[idx]], y=[panc_cy[idx] - 0.25], z=[panc_cz[idx]],
            mode='text', text=[lbl],
            textfont=dict(size=12, color=lcol, family='Inter, sans-serif'),
            hoverinfo='skip', showlegend=False,
        ))

    # ═══════════════════════════════════════════════════════
    # 8. DUODENUM — C-curve around pancreatic head
    # ═══════════════════════════════════════════════════════
    duo_path = [
        [0.45, 0.15, 0.25], [0.55, 0.20, 0.15], [0.60, 0.22, 0.05],
        [0.62, 0.20, -0.05], [0.60, 0.15, -0.15], [0.55, 0.10, -0.22],
        [0.48, 0.05, -0.28], [0.40, 0.0, -0.30], [0.32, -0.05, -0.28],
        [0.25, -0.08, -0.22], [0.20, -0.08, -0.15],
    ]
    duo_radii = np.linspace(0.08, 0.06, len(duo_path))
    duo_pts = tube_pts(duo_path, duo_radii, 12)
    fig.add_trace(mesh(duo_pts, duo_r, 0.75, 'Duodenum'))

    # ═══════════════════════════════════════════════════════
    # 9. BILE DUCT — connecting liver to duodenum
    # ═══════════════════════════════════════════════════════
    bile_path = [
        [0.50, 0.30, 0.35], [0.52, 0.28, 0.25],
        [0.55, 0.25, 0.15], [0.55, 0.22, 0.05],
        [0.52, 0.20, -0.02],
    ]
    bile_r_vals = np.linspace(0.04, 0.03, len(bile_path))
    bile_pts = tube_pts(bile_path, bile_r_vals, 10)

    bile_col = rc(bile_r, 0.8)
    bile_pts_arr = np.array(bile_pts)
    fig.add_trace(go.Mesh3d(
        x=bile_pts_arr[:,0], y=bile_pts_arr[:,1], z=bile_pts_arr[:,2],
        alphahull=4, opacity=0.8,
        color=bile_col, name='Bile Duct',
        showscale=False,
        lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3, roughness=0.3),
        hoverinfo='text',
        text=f'<b>Bile Duct</b><br>Risk: {bile_r}%',
    ))

    # ═══════════════════════════════════════════════════════
    # 10. LYMPH NODES — scattered at anatomical sites
    # ═══════════════════════════════════════════════════════
    node_sites = [
        [0.2, -0.15, 0.1], [-0.1, -0.1, 0.05], [-0.3, -0.2, 0.15],
        [0.5, -0.1, -0.15], [-0.5, -0.15, 0.2], [0.0, -0.3, -0.1],
        [0.7, 0.0, 0.6], [-0.6, 0.0, 0.8],
        [0.3, 0.1, 0.9], [-0.3, 0.1, 0.9],
    ]
    node_sizes = [5 + lymph_r/8, 4 + lymph_r/10, 5 + lymph_r/8,
                  6 + lymph_r/6, 4 + lymph_r/10, 5 + lymph_r/8,
                  3 + lymph_r/12, 3 + lymph_r/12, 4 + lymph_r/10, 4 + lymph_r/10]

    node_col = rc(lymph_r, 0.85)
    fig.add_trace(go.Scatter3d(
        x=[s[0] for s in node_sites],
        y=[s[1] for s in node_sites],
        z=[s[2] for s in node_sites],
        mode='markers',
        marker=dict(
            size=node_sizes,
            color=node_col,
            opacity=0.85,
            line=dict(color=rhex(lymph_r), width=1.5),
            symbol='circle',
        ),
        name=f'Lymph Nodes ({lymph_r}%)',
        hoverinfo='text',
        text=[f'<b>Lymph Node</b><br>Risk: {lymph_r}%' for _ in node_sites],
    ))

    # ═══════════════════════════════════════════════════════
    # 11. KIDNEYS (context)
    # ═══════════════════════════════════════════════════════
    kidney_r = max(0, int(panc_r * 0.25))
    kid_col = rc(kidney_r, 0.50)

    for side, k_label in [(-0.9, 'Left Kidney'), (0.9, 'Right Kidney')]:
        def kidney_deform(x, y, z, s=side):
            return (0.08*z, -0.05*x*s, 0)
        kid_pts = deform_ellipsoid(
            side, -0.15, 0.0, 0.13, 0.08, 0.18,
            lambda x, y, z: kidney_deform(x, y, z, side), n=300
        )
        fig.add_trace(mesh(kid_pts, kidney_r, 0.45, k_label, kid_col))

    # ═══════════════════════════════════════════════════════
    # 12. DISEASE SPREAD PATHS (glowing connections)
    # ═══════════════════════════════════════════════════════
    if panc_r > 25:
        connections = [
            ([0.2, 0.35], [-0.05, 0.05], [-0.05, 0.25], 'To Duodenum', duo_r),
            ([0.3, 0.50], [-0.1, 0.15], [0.0, 0.55], 'To Liver', liver_r),
            ([0.1, -0.2], [-0.1, 0.3], [0.0, 0.5], 'To Stomach', stom_r),
            ([-0.3, -0.8], [-0.1, -0.1], [0.3, 0.35], 'To Spleen', spl_r),
            ([0.2, 0.0], [-0.05, -0.2], [0.0, -0.1], 'Lymph spread', lymph_r),
        ]
        for xs, ys, zs, cname, cr in connections:
            opacity_path = min(0.6, 0.1 + panc_r/150)
            fig.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode='lines',
                line=dict(
                    color=rc(cr, opacity_path),
                    width=2 + cr/20,
                ),
                name=cname,
                hoverinfo='skip',
                showlegend=False,
            ))

    # ═══════════════════════════════════════════════════════
    # 13. RISK LEGEND (3D text annotations)
    # ═══════════════════════════════════════════════════════
    legend_items = [
        (-1.8, -0.7, -1.9, 'RISK SCALE', '#60a5fa', 0.7),
        (-1.8, -0.7, -2.05, '● Low <20%', '#2ecc71', 0.55),
        (-1.8, -0.7, -2.15, '● Mild 20-44%', '#f1c40f', 0.55),
        (-1.8, -0.7, -2.25, '● Moderate 45-69%', '#f39c12', 0.55),
        (-1.8, -0.7, -2.35, '● High ≥70%', '#dc3232', 0.55),
    ]
    for lx, ly, lz, lt, lc, ls in legend_items:
        fig.add_trace(go.Scatter3d(
            x=[lx], y=[ly], z=[lz],
            mode='text', text=[lt],
            textfont=dict(size=10, color=lc, family='Inter, sans-serif'),
            hoverinfo='skip', showlegend=False,
        ))

    # Title
    fig.add_trace(go.Scatter3d(
        x=[-1.8], y=[0.7], z=[-1.9],
        mode='text',
        text=['AI DIGITAL TWIN — 3D Anatomical Risk Map'],
        textfont=dict(size=13, color='#60a5fa', family='Inter, sans-serif'),
        hoverinfo='skip', showlegend=False,
    ))

    # ═══════════════════════════════════════════════════════
    # LAYOUT
    # ═══════════════════════════════════════════════════════
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showgrid=False, range=[-2.2, 2.2]),
            yaxis=dict(visible=False, showgrid=False, range=[-1.0, 1.0]),
            zaxis=dict(visible=False, showgrid=False, range=[-2.4, 2.4]),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=2.8, y=-2.2, z=1.5),
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0.3),
            ),
            aspectmode='manual',
            aspectratio=dict(x=1.3, y=0.65, z=1.2),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        showlegend=False,
    )

    return fig


def render_3d_pancreas(organ_risks):
    """Legacy detailed pancreas view — kept for zoomed-in inspection."""
    risk_pct = organ_risks["Pancreas"][0] / 100.0

    n, m = 80, 40
    s = np.linspace(0, 1, n)
    theta = np.linspace(0, 2 * np.pi, m)

    cx = 2.0 - 4.0 * s
    cy = -0.6 + 2.0 * s - 0.8 * s ** 2
    cz = 0.3 * np.sin(s * np.pi * 0.7)

    radius = np.zeros(n)
    for i, si in enumerate(s):
        if si < 0.20:
            radius[i] = 0.22 * (1 + 0.20 * np.sin(si / 0.20 * np.pi))
        elif si < 0.30:
            radius[i] = 0.22 - (si - 0.20) / 0.10 * 0.12
        elif si < 0.78:
            radius[i] = 0.10
        else:
            radius[i] = 0.10 * (1 - (si - 0.78) / 0.22 * 0.55)

    dx = np.gradient(cx, s[1] - s[0])
    dy = np.gradient(cy, s[1] - s[0])
    dz = np.gradient(cz, s[1] - s[0])
    tn = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) + 1e-12
    tx, ty, tz = dx / tn, dy / tn, dz / tn

    nx, ny, nz = -ty, tx, np.zeros_like(tx)
    nn = np.sqrt(nx ** 2 + ny ** 2) + 1e-12
    nx, ny = nx / nn, ny / nn

    bx = ty * nz - tz * ny
    by = tz * nx - tx * nz
    bz = tx * ny - ty * nx

    X = np.zeros((n, m))
    Y = np.zeros((n, m))
    Z = np.zeros((n, m))
    C = np.zeros((n, m))

    for i in range(n):
        si = s[i]
        r_i = radius[i]
        if si < 0.25:
            seg_risk = min(1.0, risk_pct * 1.15)
        elif si < 0.35:
            seg_risk = risk_pct
        elif si < 0.78:
            seg_risk = risk_pct * 0.85
        else:
            seg_risk = risk_pct * 0.60

        if risk_pct > 0.45:
            lesion = 0.30 * np.exp(-((si - 0.08) ** 2) / 0.008)
            seg_risk = min(1.0, seg_risk + lesion)

        for j in range(m):
            ct, st = np.cos(theta[j]), np.sin(theta[j])
            X[i, j] = cx[i] + r_i * (ct * nx[i] + st * bx[i])
            Y[i, j] = cy[i] + r_i * (ct * ny[i] + st * by[i])
            Z[i, j] = cz[i] + r_i * (ct * nz[i] + st * bz[i])
            C[i, j] = seg_risk

    fig = go.Figure()

    fig.add_trace(go.Surface(
        x=X, y=Y, z=Z, surfacecolor=C,
        colorscale=[
            [0.00, "#2ecc71"],
            [0.20, "#2ecc71"],
            [0.35, "#f1c40f"],
            [0.55, "#f39c12"],
            [0.75, "#e74c3c"],
            [1.00, "#c0392b"],
        ],
        showscale=False,
        opacity=0.92,
        lighting=dict(ambient=0.55, diffuse=0.75, specular=0.25, roughness=0.6),
        contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
        hoverinfo="skip",
    ))

    if risk_pct > 0.45:
        lesion_s = np.array([0.08, 0.12, 0.05, 0.15, 0.10])
        lesion_t = np.linspace(0, 2 * np.pi, len(lesion_s))
        lx = np.interp(lesion_s, s, cx) + radius[np.searchsorted(s, lesion_s)] * 1.3 * np.cos(lesion_t)
        ly = np.interp(lesion_s, s, cy) + radius[np.searchsorted(s, lesion_s)] * 1.3 * np.sin(lesion_t) * 0.6
        lz = np.interp(lesion_s, s, cz)
        fig.add_trace(go.Scatter3d(
            x=lx, y=ly, z=lz,
            mode='markers',
            marker=dict(
                size=10 + risk_pct * 16,
                color='#ff4444',
                opacity=0.25 + risk_pct * 0.30,
                line=dict(color='rgba(255,50,50,0.6)', width=2),
            ),
            name='Lesion hotspot',
            hoverinfo='skip',
        ))

    label_data = [
        (0.08, "HEAD",  risk_pct * 1.15),
        (0.45, "BODY",  risk_pct * 0.85),
        (0.90, "TAIL",  risk_pct * 0.60),
    ]
    for si, lbl, lrisk in label_data:
        idx = np.searchsorted(s, si)
        lcol = "#2ecc71" if lrisk < 0.20 else "#f1c40f" if lrisk < 0.45 else "#f39c12" if lrisk < 0.70 else "#e74c3c"
        fig.add_trace(go.Scatter3d(
            x=[cx[idx]], y=[cy[idx] - radius[idx] * 1.5], z=[cz[idx]],
            mode='text',
            text=[lbl],
            textfont=dict(size=13, color=lcol, family='Inter, sans-serif'),
            hoverinfo='skip',
            showlegend=False,
        ))

    fig.add_trace(go.Scatter3d(
        x=[cx[0] + 0.3], y=[cy[0] + 0.8], z=[cz[0]],
        mode='text',
        text=[f"Risk: {organ_risks['Pancreas'][0]}%"],
        textfont=dict(size=15, color='#ffffff', family='Inter, sans-serif'),
        hoverinfo='skip',
        showlegend=False,
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(visible=False, showgrid=False),
            zaxis=dict(visible=False, showgrid=False),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),
                up=dict(x=0, y=0, z=1),
            ),
            aspectmode='manual',
            aspectratio=dict(x=1.2, y=0.6, z=0.3),
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=0, b=0),
        height=500,
        showlegend=False,
    )

    return fig

def plot_biomarker_timeline(df):
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=["CA 19-9 (U/mL)","CEA (ng/mL)",
                                        "Bilirubin (mg/dL)","Amylase (U/L)"],
                        shared_xaxes=False)

    configs = [
        ("CA 19-9 (U/mL)",    "#e74c3c", 37,    1, 1),
        ("CEA (ng/mL)",       "#f39c12", 5,     1, 2),
        ("Bilirubin (mg/dL)", "#3b82f6", 1.2,   2, 1),
        ("Amylase (U/L)",     "#8b5cf6", 110,   2, 2),
    ]
    for col_name, color, threshold, row, col in configs:
        fig.add_trace(go.Scatter(
            x=df["Month"], y=df[col_name],
            name=col_name, mode="lines+markers",
            line=dict(color=color, width=2.5),
            marker=dict(size=6, color=color),
            hovertemplate=f"<b>{col_name}</b><br>%{{x}}<br>Value: %{{y}}<extra></extra>",
        ), row=row, col=col)
        fig.add_hline(y=threshold, line_dash="dot", line_color=color,
                      line_width=1, opacity=0.5, row=row, col=col)

    fig.update_layout(
        paper_bgcolor="#080e18",
        plot_bgcolor="#0e1e30",
        font=dict(family="Inter", color="#8bb8e0"),
        title=dict(text="12-Month Biomarker Trajectory",
                   font=dict(color="#fff", size=13)),
        showlegend=False,
        height=400,
        margin=dict(l=50, r=30, t=60, b=50),
    )
    for i in range(1,3):
        for j in range(1,3):
            fig.update_xaxes(gridcolor="#1a3555", row=i, col=j)
            fig.update_yaxes(gridcolor="#1a3555", row=i, col=j)
    return fig


# ─────────────────────────────────────────────────────────────
# ORGAN RADAR CHART
# ─────────────────────────────────────────────────────────────

def plot_organ_radar(organ_risks):
    organs = list(organ_risks.keys())
    values = [organ_risks[o][0] for o in organs]
    values_closed = values + [values[0]]
    organs_closed = organs + [organs[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values_closed, theta=organs_closed,
        fill="toself",
        fillcolor="rgba(231,76,60,0.15)",
        line=dict(color="#e74c3c", width=2),
        name="Patient Risk",
        hovertemplate="<b>%{theta}</b><br>Risk: %{r}%<extra></extra>",
    ))
    normal_vals = [5, 5, 5, 5, 5, 5, 5, 5, 5]
    fig.add_trace(go.Scatterpolar(
        r=normal_vals[:len(organs)+1], theta=organs_closed,
        fill="toself",
        fillcolor="rgba(46,204,113,0.08)",
        line=dict(color="#2ecc71", width=1.5, dash="dot"),
        name="Normal Reference",
    ))
    fig.update_layout(
        paper_bgcolor="#080e18",
        polar=dict(
            bgcolor="#0e1e30",
            radialaxis=dict(visible=True, range=[0,100],
                            gridcolor="#1a3555", tickcolor="#60a5fa",
                            tickfont=dict(size=9, color="#7a9fc5")),
            angularaxis=dict(gridcolor="#1a3555", tickcolor="#60a5fa",
                             tickfont=dict(size=10, color="#c0d8f0")),
        ),
        legend=dict(bgcolor="rgba(8,14,24,.85)", bordercolor="#1a3555",
                    borderwidth=1, font=dict(size=11)),
        title=dict(text="Organ Risk Radar",
                   font=dict(color="#fff", size=13)),
        height=380,
        margin=dict(l=40, r=40, t=55, b=40),
    )
    return fig


# ─────────────────────────────────────────────────────────────
# STANDARD PLOTS
# ─────────────────────────────────────────────────────────────

BG,PB,GC,FC,FF = "#080e18","#0e1e30","#1a3555","#8bb8e0","Inter, sans-serif"

def lay(title="", h=320):
    return dict(paper_bgcolor=BG, plot_bgcolor=PB,
                font=dict(family=FF, color=FC),
                title=dict(text=title, font=dict(color="#fff", size=13)),
                margin=dict(l=60,r=40,t=50,b=50), height=h)

def gauge_chart(prob):
    col = "#e74c3c" if prob>0.5 else "#f39c12" if prob>0.3 else "#2ecc71"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob*100,
        number=dict(suffix="%", font=dict(size=42, color="#fff", family=FF)),
        gauge=dict(
            axis=dict(range=[0,100], tickcolor=GC, tickfont=dict(color=FC,size=11)),
            bar=dict(color=col, thickness=0.2),
            bgcolor=BG, bordercolor=GC, borderwidth=1,
            steps=[dict(range=[0,30],  color="rgba(46,204,113,.10)"),
                   dict(range=[30,60], color="rgba(243,156,18,.10)"),
                   dict(range=[60,100],color="rgba(231,76,60,.10)")],
            threshold=dict(line=dict(color="#60a5fa",width=3), value=prob*100),
        ),
        title=dict(text="Cancer Probability",
                   font=dict(size=14, color=FC, family=FF)),
    ))
    fig.update_layout(**lay(h=280))
    return fig

def shap_chart(shap_vals):
    items  = sorted(shap_vals.items(), key=lambda x: abs(x[1]))
    feats, vals = [i[0] for i in items], [i[1] for i in items]
    colors = ["#e74c3c" if v>0 else "#3b82f6" for v in vals]
    fig = go.Figure(go.Bar(
        x=vals, y=feats, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.3f}" for v in vals], textposition="outside",
        textfont=dict(color=FC, size=10, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=GC, line_width=1.5)
    l = lay("Feature Attribution (SHAP)", 400)
    l["xaxis"] = dict(title="SHAP Value", color=FC, gridcolor=GC, zeroline=False)
    l["yaxis"] = dict(color=FC, tickfont=dict(size=10))
    l["margin"]["l"] = 180
    fig.update_layout(**l)
    return fig

def tx_chart(resp):
    txs,probs = list(resp.keys()), [v*100 for v in resp.values()]
    colors = ["#2ecc71" if p>60 else "#f39c12" if p>40 else "#e74c3c" for p in probs]
    fig = go.Figure(go.Bar(
        x=txs, y=probs,
        marker=dict(color=colors, opacity=0.85, line=dict(width=0)),
        text=[f"{p:.0f}%" for p in probs], textposition="outside",
        textfont=dict(color="#fff", size=12),
    ))
    fig.add_hline(y=50, line_dash="dot", line_color="#4a6fa8",
                  annotation_text="50% threshold", annotation_font_color=FC)
    l = lay("Treatment Response", 330)
    l["yaxis"] = dict(title="Response Probability (%)", range=[0,108], color=FC, gridcolor=GC)
    l["xaxis"] = dict(color=FC, tickfont=dict(size=10))
    fig.update_layout(**l)
    return fig

def km_chart(surv):
    t   = np.linspace(0,36,120)
    wb  = lambda lam,t,k=1.15: np.exp(-(lam*t)**k)
    lam = np.log(2)/max(surv["median_months"],1)
    fig = go.Figure()
    for lbl,(l_,col,dash) in [
        ("Low Risk (population)",   (0.018,"#2ecc71","dot")),
        ("Medium Risk (population)",(0.048,"#f39c12","dot")),
        ("High Risk (population)",  (0.120,"#e74c3c","dot")),
    ]:
        fig.add_trace(go.Scatter(x=t,y=wb(l_,t)*100,mode="lines",name=lbl,
                                  line=dict(color=col,width=1.5,dash=dash),opacity=0.45))
    fig.add_trace(go.Scatter(
        x=t, y=wb(lam,t)*100, mode="lines",
        name=f"<b>This Patient ({surv['risk_group']})</b>",
        line=dict(color=surv["risk_color"],width=3),
    ))
    fig.add_vline(x=surv["median_months"],line_dash="dash",line_color="#60a5fa",line_width=1.5,
                  annotation_text=f"Median: {surv['median_months']}m",
                  annotation_font_color="#60a5fa",annotation_font_size=11)
    l = lay("Projected Survival Curve", 380)
    l["xaxis"] = dict(title="Time (Months)", color=FC, gridcolor=GC)
    l["yaxis"] = dict(title="Survival Probability (%)", range=[0,105], color=FC, gridcolor=GC)
    l["legend"]= dict(bgcolor="rgba(8,14,24,.85)",bordercolor=GC,borderwidth=1,font=dict(size=11))
    fig.update_layout(**l)
    return fig


# ─────────────────────────────────────────────────────────────
# CSV PARSING
# ─────────────────────────────────────────────────────────────

def parse_csv(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        if df.empty:
            return None
        row = df.iloc[0]

        def get(keys, default, cast=float):
            for k in keys:
                for col in df.columns:
                    if col.strip().lower() == k.lower():
                        try: return cast(row[col])
                        except: pass
            return default

        return {
            "patient_name":   str(row.get("patient_id", row.get("Patient ID","PT-CSV-001"))),
            "age":            int(get(["age"],65,float)),
            "sex":            str(row.get("sex", row.get("Sex","Male"))),
            "ca199":          get(["ca199","ca 19-9","ca19-9"],45.0),
            "cea":            get(["cea"],3.5),
            "bilirubin":      get(["bilirubin","total bilirubin"],0.8),
            "amylase":        get(["amylase"],95.0),
            "weight_loss_pct":get(["weight_loss_pct","weight loss (%)"],5.0),
            "smoking":        str(row.get("smoking","Never")),
            "alcohol":        str(row.get("alcohol","None / Minimal")),
            "diabetes":       str(row.get("diabetes","None")),
            "family_history": bool(int(get(["family_history"],0,float))),
            "pancreatitis":   bool(int(get(["pancreatitis"],0,float))),
            "genetic":        bool(int(get(["genetic"],0,float))),
        }
    except Exception as e:
        st.sidebar.error(f"CSV parse error: {e}")
        return None


def generate_template_csv():
    d = {
        "patient_id":"PT-2024-001","age":65,"sex":"Male",
        "ca199":45.0,"cea":3.5,"bilirubin":0.8,"amylase":95,
        "weight_loss_pct":5,"smoking":"Never",
        "alcohol":"None / Minimal","diabetes":"None",
        "family_history":0,"pancreatitis":0,"genetic":0,
    }
    return pd.DataFrame([d]).to_csv(index=False)


# ─────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None

# Digital Twin engine state (persists across interactions)
if "dt_engine" not in st.session_state:
    st.session_state.dt_engine = DigitalTwinEngine()
if "dt_loaded" not in st.session_state:
    st.session_state.dt_loaded = False
if "dt_mode" not in st.session_state:
    st.session_state.dt_mode = "Risk Heatmap"
if "dt_progress_months" not in st.session_state:
    st.session_state.dt_progress_months = 0


# ─────────────────────────────────────────────────────────────
# HERO BANNER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <div class='hero-badge'>TCGA-PAAD · GTEx · Clinical Validation</div>
  <div class='hero-title'>ExAI-<span>PanC</span></div>
  <p class='hero-sub'>Explainable AI for Pancreatic Cancer — Detection · Staging · Treatment · Survival · Clinical Digital Twin</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:.6rem 0 1rem;'>
      <div style='font-size:1.5rem;'>🧬</div>
      <div style='font-weight:700;color:#fff;font-size:.95rem;'>ExAI-PanC</div>
      <div style='font-size:.65rem;color:#2a4a6a;letter-spacing:1px;text-transform:uppercase;'>Clinical Decision Support</div>
    </div>
    <hr style='border-color:#1a3555;margin:0 0 .8rem;'>
    """, unsafe_allow_html=True)

    # CSV Upload
    st.markdown("### 📂 Import")
    st.markdown("<div class='upload-zone'>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"],
                                 label_visibility="collapsed",
                                 help="Upload a CSV with patient data to auto-fill all fields")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded is not None:
        parsed = parse_csv(uploaded)
        if parsed:
            st.session_state.csv_data = parsed
            st.success("✅ CSV loaded")

    st.download_button("📥 Download Template", data=generate_template_csv(),
                       file_name="ExAI-PanC_template.csv", mime="text/csv",
                       use_container_width=True)

    st.markdown("<hr style='border-color:#1a3555;margin:.6rem 0;'>", unsafe_allow_html=True)

    D = st.session_state.csv_data or {}

    # ── PATIENT ──
    st.markdown("<div class='side-sec'>Patient</div>", unsafe_allow_html=True)
    patient_name = st.text_input("ID / Name", value=D.get("patient_name","PT-2024-001"))
    age   = st.slider("Age", 30, 90, int(D.get("age",65)))
    sex   = st.selectbox("Sex", ["Male","Female"],
                          index=0 if D.get("sex","Male")=="Male" else 1)

    # ── BIOMARKERS ──
    st.markdown("<div class='side-sec'>Biomarkers</div>", unsafe_allow_html=True)
    ca199        = st.number_input("CA 19-9 (U/mL)", 0.0,10000.0, float(D.get("ca199",45.0)),   5.0)
    cea          = st.number_input("CEA (ng/mL)",     0.0,500.0,   float(D.get("cea",3.5)),       0.5)
    bilirubin    = st.number_input("Bilirubin (mg/dL)",0.0,30.0,  float(D.get("bilirubin",0.8)), 0.1)
    amylase      = st.number_input("Amylase (U/L)",   0,  3000,   int(D.get("amylase",95)),      10)
    weight_loss_pct = st.slider("Weight Loss (%)", 0, 30, int(D.get("weight_loss_pct",5)))

    # ── RISK FACTORS ──
    st.markdown("<div class='side-sec'>Risk Factors</div>", unsafe_allow_html=True)
    smoke_opts   = ["Never","Former","Current"]
    alcohol_opts = ["None / Minimal","Moderate (<14/week)","Heavy (>14/week)"]
    dm_opts      = ["None","New-onset (<2 years)","Long-standing"]

    smoking        = st.selectbox("Smoking", smoke_opts, index=0)
    alcohol        = st.selectbox("Alcohol", alcohol_opts, index=0)
    diabetes       = st.selectbox("Diabetes", dm_opts, index=0)
    family_history = st.checkbox("Family history of PDAC",    value=False)
    pancreatitis   = st.checkbox("Chronic pancreatitis",       value=False)
    genetic        = st.checkbox("BRCA1/2 or PALB2 variant",  value=False)

    st.markdown("<hr style='border-color:#1a3555;margin:.8rem 0;'>", unsafe_allow_html=True)
    run_btn = st.button("⚡  Run Prediction")
    st.markdown("""<div style='font-size:.65rem;color:#1a3a5a;text-align:center;margin-top:.6rem;line-height:1.6;'>
    AUC 0.999 · C-index 0.857<br>
    <span style='color:#0f2a42;'>⚠ Research use only</span></div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# LANDING PAGE (before prediction)
# ─────────────────────────────────────────────────────────────
if not run_btn:
    c1,c2,c3,c4 = st.columns(4)
    for col,(v,l,s) in zip([c1,c2,c3,c4],[
        ("0.999","AUC-ROC","TCGA vs GTEx — 5-Fold CV"),
        ("99.3%", "Accuracy","Detection classifier"),
        ("0.857","C-index","Survival model"),
        ("138", "Patients","Clinical training cohort"),
    ]):
        with col:
            st.markdown(f"<div class='card'><div class='cv' style='color:#60a5fa;'>{v}</div>"
                        f"<div class='cl'>{l}</div><div class='cs'>{s}</div></div>",
                        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1.6,1])
    with col_a:
        st.markdown("""<div class='info-box'>
        <b>Getting Started:</b><br>
        📂 <b>CSV Import</b> — Upload a patient CSV to auto-fill all fields.<br>
        🧬 <b>AI Digital Twin</b> — Colour-coded body map showing organ-level risk.<br>
        📈 <b>Biomarker Timeline</b> — 12-month trajectory simulation.<br>
        🫀 <b>Organ Radar</b> — Multi-organ risk visualisation.<br><br>
        Fill in the patient panel on the left (or upload CSV), then click <b>Run Prediction</b>.
        </div>""", unsafe_allow_html=True)
    with col_b:
        st.markdown("""<div class='card' style='text-align:left;padding:1rem;'>
        <div style='font-size:.7rem;color:#60a5fa;text-transform:uppercase;letter-spacing:1px;margin-bottom:.6rem;font-weight:600;'>Pipeline</div>
        <div style='font-size:.78rem;color:#7a9fc5;line-height:2;'>
        🧬 Transcriptomic Embedding (500 genes → 64-dim)<br>
        🔬 Cancer Detection (AUC 0.999)<br>
        📋 Clinical Stage Prediction<br>
        📈 Survival Analysis (C-index 0.857)<br>
        💡 SHAP Feature Attribution<br>
        🫀 AI Digital Twin — Organ Risk Map
        </div></div>""", unsafe_allow_html=True)

    st.markdown("""<br><div class='warn-box'>⚠ <b>Research Use Only.</b>
    ExAI-PanC is a research prototype trained on TCGA-PAAD, GTEx Pancreas, and clinical validation cohorts.
    Not approved for clinical use. All outputs must be interpreted by qualified medical professionals.</div>""",
                unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
# RUN PREDICTIONS
# ─────────────────────────────────────────────────────────────
inputs = dict(
    age=age, sex=sex,
    ca199=ca199, cea=cea, bilirubin=bilirubin,
    amylase=amylase, weight_loss_pct=weight_loss_pct,
    smoking=smoking, alcohol=alcohol, diabetes=diabetes,
    family_history=family_history, pancreatitis=pancreatitis, genetic=genetic,
)

with st.spinner("Running analysis pipeline…"):
    result = run_full_prediction(inputs)

    prob         = result['prob']
    stage_label  = result['stage_label']
    stage_col    = result['stage_col']
    stage_detail = result['stage_detail']
    treatment    = result['treatment']
    surv         = result['survival']
    shap_vals    = result['shap_vals']
    organ_risks  = result['organ_risks']
    bio_timeline = result['bio_timeline']
    best_tx      = result['best_tx']
    sorted_sv    = result['sorted_sv']


# ─────────────────────────────────────────────────────────────
# TOP RESULT BANNER
# ─────────────────────────────────────────────────────────────
if prob >= 0.50:
    st.markdown(f"""<div class='res-hi'>
    <div class='res-title' style='color:#e74c3c;'>🔴 HIGH RISK DETECTED</div>
    <div style='font-size:.85rem;opacity:.85;'>P(Cancer) = <b>{prob*100:.1f}%</b> &nbsp;|&nbsp;
    Stage: <b style='color:{stage_col};'>{stage_label}</b> &nbsp;|&nbsp;
    Risk: <b style='color:{surv["risk_color"]};'>{surv["risk_group"]}</b> &nbsp;|&nbsp;
    Recommended: <b style='color:#2ecc71;'>{best_tx}</b></div></div>""", unsafe_allow_html=True)
else:
    st.markdown(f"""<div class='res-lo'>
    <div class='res-title' style='color:#2ecc71;'>🟢 LOW RISK — MONITORING RECOMMENDED</div>
    <div style='font-size:.85rem;opacity:.85;'>P(Cancer) = <b>{prob*100:.1f}%</b> &nbsp;|&nbsp;
    Stage: <b style='color:{stage_col};'>{stage_label}</b> &nbsp;|&nbsp;
    Risk: <b style='color:{surv["risk_color"]};'>{surv["risk_group"]}</b></div></div>""",
                unsafe_allow_html=True)

# Summary metrics
c1,c2,c3,c4,c5 = st.columns(5)
for col,(v,c,l,s) in zip([c1,c2,c3,c4,c5],[
    (f"{prob*100:.1f}%",            "#e74c3c" if prob>0.5 else "#2ecc71","Cancer Prob.","Detection"),
    (stage_label,                    stage_col,                            "Stage","Staging"),
    (surv["risk_group"],             surv["risk_color"],                   "Risk Group","Survival"),
    (f"{surv['median_months']}m",    surv["risk_color"],                   "Median Survival","Survival"),
    (f"{surv['p1y']*100:.0f}%",      surv["risk_color"],                   "1-Year Survival","Survival"),
]):
    with col:
        st.markdown(f"<div class='card'><div class='cv' style='color:{c};font-size:1.4rem;'>{v}</div>"
                    f"<div class='cl'>{l}</div><div class='cs'>{s}</div></div>",
                    unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
t1,t2,t3,t4,t5,t6 = st.tabs([
    "🫀 Clinical Digital Twin",
    "📊 Detection",
    "🎯 Stage & Treatment",
    "📈 Survival",
    "💡 SHAP",
    "📋 Report",
])


# ═════════════════════════════════════════════════════════════
# TAB 1: CLINICAL DIGITAL TWIN — Modular next-gen twin
# ═════════════════════════════════════════════════════════════
with t1:
    dt_engine = st.session_state.dt_engine
    if not st.session_state.dt_loaded:
        dt_engine.load_patient(inputs, result)
        st.session_state.dt_loaded = True
    else:
        dt_engine.patient_data = inputs
        dt_engine.predictions = result

    # ── Callbacks (no st.rerun) ──
    def _on_dt_mode_change():
        st.session_state.dt_mode = st.session_state.dt_mode_selector
        if st.session_state.dt_mode != "Disease Progression":
            st.session_state.dt_progress_months = 0
            st.session_state.dt_engine.update_progression(0)

    def _on_dt_progress_change():
        st.session_state.dt_progress_months = st.session_state.dt_progress_slider
        st.session_state.dt_engine.update_progression(st.session_state.dt_progress_months)

    def _on_dt_shap_change():
        feat = st.session_state.dt_shap_selector
        if feat:
            st.session_state.dt_engine.highlight_shap_feature(feat)
        else:
            st.session_state.dt_engine.reset_highlight()

    def _on_dt_tx_change():
        tx = st.session_state.dt_tx_selector
        treat = st.session_state.dt_engine.predictions.get("treatment", {})
        resp = treat.get(tx, 0.0) if tx != "None" else 0.0
        st.session_state.dt_engine.apply_treatment(tx, resp)

    # ── Mode selector ──
    mode_names = list(RENDER_MODES.keys())
    col_modes, col_info = st.columns([3, 1])
    with col_modes:
        st.segmented_control(
            "Visualisation Mode", mode_names,
            default=st.session_state.dt_mode,
            key="dt_mode_selector",
            selection_mode="single",
            label_visibility="collapsed",
            on_change=_on_dt_mode_change,
        )
    with col_info:
        st.html(
            f"<div style='text-align:right;font-size:11px;color:#60a5fa;padding-top:6px;'>"
            f"Mode: <b>{st.session_state.dt_mode}</b></div>"
        )

    # ── Main layout ──
    dt_main, dt_panel = st.columns([2.0, 1.0])

    with dt_main:
        # Progression slider
        if st.session_state.dt_mode == "Disease Progression":
            st.slider(
                "Progression Timeline (months)",
                0, 36, st.session_state.dt_progress_months, 1,
                key="dt_progress_slider",
                label_visibility="collapsed",
                on_change=_on_dt_progress_change,
            )

        # Treatment selector
        if st.session_state.dt_mode == "Treatment Response":
            tx_options = list(result.get("treatment", {}).keys()) + ["None"]
            st.selectbox(
                "Select Treatment", tx_options, index=0,
                key="dt_tx_selector",
                on_change=_on_dt_tx_change,
            )

        # SHAP highlight
        shap_feats = [""] + [k for k in result.get("shap_vals", {}).keys()]
        st.selectbox(
            "Highlight SHAP Feature", shap_feats, index=0,
            key="dt_shap_selector",
            label_visibility="collapsed",
            placeholder="Select SHAP feature to highlight...",
            on_change=_on_dt_shap_change,
        )

        # Render (cached in session state for current mode)
        cache_key = f"dt_fig_{st.session_state.dt_mode}_{st.session_state.dt_progress_months}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = dt_engine.display(mode=st.session_state.dt_mode)
        fig_twin = st.session_state[cache_key]
        st.plotly_chart(fig_twin, use_container_width=True, config={"displayModeBar": False})

    with dt_panel:
        st.html(patient_panel_html(dt_engine))
        organ_names = list(dt_engine.organs.keys())
        sel_org = st.selectbox("Organ Details", organ_names, index=0,
                                label_visibility="collapsed",
                                placeholder="Select organ for details...")
        if sel_org and sel_org in dt_engine.organs:
            st.html(organ_hover_html(dt_engine.organs[sel_org]))

    # ── Legacy ──
    with st.expander("Legacy Visualizations — 3D Pancreas, Organ Radar, Body Map"):
        leg_col1, leg_col2 = st.columns(2)
        with leg_col1:
            st.html("<div style='text-align:center;font-size:.8rem;color:#60a5fa;font-weight:600;margin-bottom:.4rem;'>Detailed 3D Pancreas</div>")
            st.plotly_chart(render_3d_pancreas(organ_risks), use_container_width=True, config={"displayModeBar": False})
        with leg_col2:
            st.html("<div style='text-align:center;font-size:.8rem;color:#60a5fa;font-weight:600;margin-bottom:.4rem;'>Multi-Organ Radar</div>")
            st.plotly_chart(plot_organ_radar(organ_risks), use_container_width=True, config={"displayModeBar": False})
        st.html("<div style='text-align:center;font-size:.8rem;color:#60a5fa;font-weight:600;margin-bottom:.4rem;'>Anatomical Overview (2D Body Map)</div>")
        svg = render_ai_twin(organ_risks)
        st.html(svg)

    # ── Biomarker Timeline ──
    st.html("<div style='font-size:.85rem;font-weight:600;color:#60a5fa;border-left:3px solid #60a5fa;padding-left:10px;margin:1.3rem 0 .7rem;text-transform:uppercase;letter-spacing:.5px;'>12-Month Biomarker Trajectory</div>")
    st.html("<div style='background:rgba(59,130,246,.06);border:1px solid rgba(59,130,246,.2);border-radius:8px;padding:.8rem 1.1rem;font-size:.82rem;color:#8bb8e0;line-height:1.6;margin-bottom:.8rem;'>Projected trajectory based on current values and predicted risk. Dotted lines = clinical thresholds. This is a simulation — not a clinical prognosis.</div>")
    st.plotly_chart(plot_biomarker_timeline(bio_timeline), use_container_width=True, config={"displayModeBar": False})
    with st.expander("View raw data"):
        st.dataframe(bio_timeline, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
# TAB 2: DETECTION — Big gauge + biomarker panel
# ═════════════════════════════════════════════════════════════
with t2:
    st.markdown("<div class='sec-hdr'>Cancer Detection</div>", unsafe_allow_html=True)

    det_col, bio_col = st.columns([1.3, 1])

    with det_col:
        st.plotly_chart(gauge_chart(prob), use_container_width=True,
                        config={"displayModeBar": False})
        conf  = "Very High" if abs(prob-0.5)>0.3 else "Moderate" if abs(prob-0.5)>0.15 else "Low"
        cc    = "#2ecc71" if conf=="Very High" else "#f39c12" if conf=="Moderate" else "#e74c3c"
        st.markdown(f"<div style='text-align:center;font-size:.8rem;color:#5a7a9a;'>Confidence: "
                    f"<span style='color:{cc};font-weight:600;'>{conf}</span></div>",
                    unsafe_allow_html=True)

    with bio_col:
        st.markdown("<div class='sec-hdr'>Biomarker Status</div>", unsafe_allow_html=True)
        bm_rows = [
            ("CA 19-9 (U/mL)",  ca199,         37,  False),
            ("CEA (ng/mL)",     cea,            5,   False),
            ("Bilirubin mg/dL", bilirubin,      1.2, False),
            ("Weight Loss (%)", weight_loss_pct,5,   False),
        ]
        for name, val, thresh, flip in bm_rows:
            hi   = (val<thresh) if flip else (val>thresh)
            col_ = "#e74c3c" if hi else "#2ecc71"
            stat = "⚠ Elevated" if hi else "✓ Normal"
            st.markdown(f"""<div class='bm-row'>
            <span style='font-size:.8rem;color:#7a9fc5;'>{name}</span>
            <div><span class='mono'>{val}</span>
            <span style='margin-left:.6rem;font-size:.72rem;color:{col_};font-weight:600;'>{stat}</span>
            </div></div>""", unsafe_allow_html=True)

        st.markdown("<div class='sec-hdr'>Risk Factors</div>", unsafe_allow_html=True)
        risks = []
        if ca199>37:        risks.append(f"CA 19-9 elevated ({ca199:.0f} U/mL)")
        if bilirubin>1.2:   risks.append("Hyperbilirubinaemia")
        if smoking!="Never":risks.append(f"Smoking ({smoking})")
        if diabetes!="None":risks.append(f"Diabetes ({diabetes})")
        if family_history:  risks.append("Family history PDAC")
        if pancreatitis:    risks.append("Chronic pancreatitis")
        if genetic:         risks.append("BRCA/PALB2 variant")

        if risks:
            for r in risks:
                st.markdown(f"<div style='font-size:.8rem;color:#f39c12;padding:2px 0;'>⚠ {r}</div>",
                            unsafe_allow_html=True)
        else:
            st.markdown("<div style='font-size:.8rem;color:#2ecc71;'>✓ No major risk factors</div>",
                        unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# TAB 3: STAGE & TREATMENT
# ═════════════════════════════════════════════════════════════
with t3:
    s_col, tx_col = st.columns([1, 1.5])
    with s_col:
        st.markdown("<div class='sec-hdr'>Stage Prediction</div>", unsafe_allow_html=True)
        st.markdown(f"""<div style='background:linear-gradient(145deg,#0e1e30,#0b1828);
                        border:2px solid {stage_col};border-radius:12px;
                        padding:1.6rem;text-align:center;margin-bottom:.8rem;'>
          <div style='font-size:2.4rem;font-weight:800;color:{stage_col};'>{stage_label}</div>
          <div style='font-size:.82rem;color:#7a9fc5;margin-top:.3rem;'>{stage_detail}</div>
        </div>""", unsafe_allow_html=True)
        stages = ["Stage I","Stage II","Stage III","Stage IV"]
        cols_s = st.columns(4)
        for i,(cs,sn) in enumerate(zip(cols_s,stages)):
            sc2 = ["#2ecc71","#f39c12","#e67e22","#e74c3c"][i]
            act = (sn==stage_label)
            with cs:
                st.markdown(f"""<div style='text-align:center;padding:.3rem .1rem;
                background:{"rgba(59,130,246,.08)" if act else "transparent"};
                border:{"2px solid "+sc2 if act else "1px solid "+sc2};
                border-radius:6px;opacity:{"1" if act else "0.35"};'>
                <div style='font-size:.65rem;color:{sc2};font-weight:{"700" if act else "400"};'>{sn}</div>
                </div>""", unsafe_allow_html=True)
    with tx_col:
        st.markdown("<div class='sec-hdr'>Treatment Response</div>", unsafe_allow_html=True)
        st.plotly_chart(tx_chart(treatment), use_container_width=True,
                        config={"displayModeBar": False})
        bc = "#2ecc71" if treatment[best_tx]>0.6 else "#f39c12"
        st.markdown(f"""<div style='background:rgba(46,204,113,.07);border:1px solid rgba(46,204,113,.25);
        border-radius:8px;padding:.8rem 1.1rem;'>
        <div style='font-size:.68rem;color:#2ecc71;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:.25rem;'>Recommended</div>
        <div style='font-size:.95rem;color:#fff;font-weight:600;'>{best_tx}</div>
        <div style='font-size:.8rem;color:#7a9fc5;'>Response probability: <span style='color:{bc};font-weight:600;'>{treatment[best_tx]*100:.0f}%</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sec-hdr'>All Treatment Options</div>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame([{
        "Treatment":k,
        "Response Probability":f"{v*100:.1f}%",
        "Recommendation":"✅ First-line" if k==best_tx else "🔄 Alternative" if v>0.40 else "⚠ Low response"
    } for k,v in treatment.items()]), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
# TAB 4: SURVIVAL
# ═════════════════════════════════════════════════════════════
with t4:
    st.markdown("<div class='sec-hdr'>Survival Analysis</div>", unsafe_allow_html=True)
    km_col, sv_col = st.columns([2,1])
    with km_col:
        st.plotly_chart(km_chart(surv), use_container_width=True,
                        config={"displayModeBar": False})
    with sv_col:
        rc = surv["risk_color"]
        st.markdown(f"""<div class='card' style='margin-bottom:.7rem;'>
        <div style='font-size:.68rem;color:#60a5fa;text-transform:uppercase;letter-spacing:1px;
                    margin-bottom:.4rem;font-weight:600;'>Risk Score</div>
        <div style='font-size:2rem;font-weight:800;color:{rc};'>{surv['risk_score']:+.3f}</div>
        <div style='font-size:.7rem;color:#2a4a6a;'>Negative=lower · Positive=higher hazard</div>
        </div>""", unsafe_allow_html=True)
        for lbl, val in [("Risk Group",surv["risk_group"]),("Median Survival",f"{surv['median_months']} months"),
                          ("1-Year Survival",f"{surv['p1y']*100:.0f}%"),("2-Year Survival",f"{surv['p2y']*100:.0f}%")]:
            st.markdown(f"""<div class='bm-row'>
            <span style='font-size:.8rem;color:#7a9fc5;'>{lbl}</span>
            <span style='color:{rc};font-weight:600;font-size:.85rem;'>{val}</span></div>""",
                        unsafe_allow_html=True)

    st.markdown("<div class='sec-hdr'>Survival at Key Timepoints</div>", unsafe_allow_html=True)
    lam_ = np.log(2)/max(surv["median_months"],1)
    pts  = [3,6,12,18,24,36]
    sp   = [float(np.exp(-(lam_*t)**1.15)) for t in pts]
    st.dataframe(pd.DataFrame({"Timepoint":[f"{t} months" for t in pts],
                                "Survival Probability":[f"{s*100:.1f}%" for s in sp],
                                "Risk Group":[surv["risk_group"]]*6}),
                 use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════
# TAB 5: SHAP
# ═════════════════════════════════════════════════════════════
with t5:
    st.markdown("<div class='sec-hdr'>Feature Attribution (SHAP)</div>", unsafe_allow_html=True)
    st.markdown("""<div class='info-box'><b>Red</b> pushes toward cancer. <b>Blue</b> pushes toward normal.
    Bar length = magnitude of influence. Based on Shapley values.</div>""",
                unsafe_allow_html=True)
    w_col, r_col = st.columns([1.6,1])
    with w_col:
        st.plotly_chart(shap_chart(shap_vals), use_container_width=True,
                        config={"displayModeBar": False})
    with r_col:
        st.markdown("<div class='sec-hdr'>Top Attributions</div>", unsafe_allow_html=True)
        total_mag = sum(abs(v) for v in shap_vals.values()) or 1
        for i,(f,v) in enumerate(sorted_sv[:10]):
            pct = abs(v)/total_mag*100
            dc  = "#e74c3c" if v>0 else "#3b82f6"
            st.markdown(f"""<div style='padding:.45rem .6rem;margin-bottom:.25rem;
            background:#0e1e30;border-radius:6px;border:1px solid #1a3555;'>
            <div style='display:flex;justify-content:space-between;margin-bottom:.2rem;'>
            <span style='font-size:.78rem;color:#8bb8e0;'>{i+1}. {f}</span>
            <span style='font-size:.72rem;color:{dc};font-weight:600;'>{v:+.4f}</span></div>
            <div style='background:#1a3555;border-radius:3px;height:3.5px;'>
            <div style='height:100%;width:{min(pct*3,100):.0f}%;background:{dc};border-radius:3px;'></div>
            </div></div>""", unsafe_allow_html=True)

    top_p = ", ".join([f"{f} ({v:+.3f})" for f,v in sorted_sv if v>0][:2]) or "none"
    top_n = ", ".join([f"{f} ({v:+.3f})" for f,v in sorted_sv if v<0][:2]) or "none"
    st.markdown(f"""<div class='info-box' style='margin-top:.8rem;'>
    <b>Explanation for {patient_name}:</b> Cancer probability <b>{prob*100:.1f}%</b>.<br>
    Cancer-pushing: <span style='color:#e74c3c;'>{top_p}</span>.<br>
    Normal-pushing: <span style='color:#3b82f6;'>{top_n}</span>.<br>
    Primary driver: <b>{sorted_sv[0][0]}</b> ({abs(sorted_sv[0][1])/total_mag*100:.0f}% of attribution).
    </div>""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════
# TAB 6: REPORT
# ═════════════════════════════════════════════════════════════
with t6:
    st.markdown("<div class='sec-hdr'>Clinical Decision Report</div>", unsafe_allow_html=True)
    now  = datetime.now().strftime("%d %B %Y, %H:%M")
    rl   = "HIGH" if prob>0.65 else "MODERATE" if prob>0.35 else "LOW"
    ho_name = max(organ_risks, key=lambda x: organ_risks[x][0])

    st.markdown(f"""
    <div style='background:#0e1e30;border:1px solid #1a3555;border-radius:12px;padding:1.8rem;'>
      <div style='border-bottom:1px solid #1a3555;padding-bottom:.8rem;margin-bottom:1.2rem;'>
        <div style='font-size:1.1rem;font-weight:700;color:#fff;'>ExAI-PanC — Clinical Decision Report</div>
        <div style='font-size:.72rem;color:#2a4a6a;'>{now} · Research Framework</div>
      </div>
      <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:1rem;margin-bottom:1.2rem;'>
        <div>
          <div style='font-size:.68rem;color:#60a5fa;text-transform:uppercase;letter-spacing:1px;margin-bottom:.4rem;font-weight:600;'>Patient</div>
          <div style='font-size:.8rem;color:#7a9fc5;line-height:1.8;'>
            ID: <b style='color:#e0e6f0;'>{patient_name}</b><br>
            Age/Sex: <b style='color:#e0e6f0;'>{age}y / {sex}</b>
          </div>
        </div>
        <div>
          <div style='font-size:.68rem;color:#60a5fa;text-transform:uppercase;letter-spacing:1px;margin-bottom:.4rem;font-weight:600;'>Predictions</div>
          <div style='font-size:.8rem;color:#7a9fc5;line-height:1.8;'>
            Cancer P: <b style='color:{"#e74c3c" if prob>0.5 else "#2ecc71"};'>{prob*100:.1f}% ({rl})</b><br>
            Stage: <b style='color:{stage_col};'>{stage_label}</b><br>
            Survival: <b style='color:{surv["risk_color"]};'>{surv["risk_group"]} · {surv["median_months"]}m</b><br>
            Recommend: <b style='color:#2ecc71;'>{best_tx}</b>
          </div>
        </div>
        <div>
          <div style='font-size:.68rem;color:#60a5fa;text-transform:uppercase;letter-spacing:1px;margin-bottom:.4rem;font-weight:600;'>Highest Risk Organ</div>
          <div style='font-size:1.3rem;font-weight:800;color:{organ_risks[ho_name][1]};'>{ho_name}</div>
          <div style='font-size:.8rem;color:#7a9fc5;'>{organ_risks[ho_name][0]}% risk</div>
          <div style='font-size:.7rem;color:#3a5a7a;margin-top:.2rem;'>{organ_risks[ho_name][2]}</div>
        </div>
      </div>
      <div style='background:#080e18;border-radius:8px;padding:.8rem;margin-bottom:.8rem;'>
        <div style='font-size:.68rem;color:#60a5fa;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:.4rem;'>Biomarkers</div>
        <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:.3rem;font-size:.78rem;color:#7a9fc5;'>
          <div>CA 19-9: <b style='color:#e0e6f0;'>{ca199}</b></div>
          <div>CEA: <b style='color:#e0e6f0;'>{cea}</b></div>
          <div>Bilirubin: <b style='color:#e0e6f0;'>{bilirubin}</b></div>
        </div>
      </div>
      <div style='background:#080e18;border-radius:8px;padding:.8rem;margin-bottom:.8rem;'>
        <div style='font-size:.68rem;color:#60a5fa;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:.4rem;'>Top SHAP Attributions</div>
        {"".join([f'<div style="font-size:.78rem;color:#7a9fc5;padding:2px 0;">{f}: <span style="color:{"#e74c3c" if v>0 else "#3b82f6"};font-weight:600;">{v:+.4f}</span></div>' for f,v in sorted_sv[:6]])}
      </div>
      <div style='background:#080e18;border-radius:8px;padding:.8rem;margin-bottom:.8rem;'>
        <div style='font-size:.68rem;color:#60a5fa;font-weight:600;text-transform:uppercase;letter-spacing:1px;margin-bottom:.4rem;'>Organ Risk Summary</div>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:.3rem;font-size:.75rem;color:#7a9fc5;'>
          {"".join([f'<div>{o}: <span style="color:{r[1]};font-weight:600;">{r[0]}%</span></div>' for o,r in sorted(organ_risks.items(),key=lambda x:x[1][0],reverse=True)])}
        </div>
      </div>
      <div style='font-size:.7rem;color:#1a3a5a;border-top:1px solid #1a3555;padding-top:.7rem;line-height:1.5;'>
        ⚠ Research prototype only. Not approved for clinical use. Requires qualified oncologist review.
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    d1,d2,d3 = st.columns(3)

    report_txt = f"""ExAI-PanC Report — {patient_name} — {now}
{'='*60}
Cancer Probability : {prob*100:.1f}% ({rl})
Stage              : {stage_label}
Risk Group         : {surv['risk_group']}
Median Survival    : {surv['median_months']} months
1-Year Survival    : {surv['p1y']*100:.0f}%
Recommended Tx     : {best_tx}
Risk Score         : {surv['risk_score']:+.3f}
Highest Risk Organ : {ho_name} ({organ_risks[ho_name][0]}%)

BIOMARKERS
CA 19-9   : {ca199} U/mL
CEA       : {cea} ng/mL
Bilirubin : {bilirubin} mg/dL

ORGAN RISK SCORES (AI Twin)
{chr(10).join([f'{o}: {r[0]}%' for o,r in sorted(organ_risks.items(),key=lambda x:x[1][0],reverse=True)])}

TOP SHAP ATTRIBUTIONS
{chr(10).join([f'{f}: {v:+.4f}' for f,v in sorted_sv[:8]])}

DISCLAIMER: Research use only. Not for clinical decision-making without physician review.
"""
    with d1:
        st.download_button("📥 Text Report", data=report_txt,
                           file_name=f"ExAI-PanC_{patient_name}_{datetime.now().strftime('%Y%m%d')}.txt",
                           mime="text/plain", use_container_width=True)
    with d2:
        csv_row = {
            "Patient":patient_name,"Date":now,
            "Cancer_Prob":f"{prob*100:.2f}%","Stage":stage_label,
            "Risk_Group":surv["risk_group"],"Median_Survival":surv["median_months"],
            "Best_Treatment":best_tx,"CA199":ca199,"CEA":cea,"Bilirubin":bilirubin,
            "Highest_Risk_Organ":ho_name,
            **{f"Organ_{o.replace(' ','_')}":r[0] for o,r in organ_risks.items()},
            **{f"SHAP_{k.replace(' ','_')}":v for k,v in list(shap_vals.items())[:10]},
        }
        st.download_button("📊 CSV Export",
                           data=pd.DataFrame([csv_row]).to_csv(index=False),
                           file_name=f"ExAI-PanC_{patient_name}.csv",
                           mime="text/csv", use_container_width=True)
    with d3:
        organ_df = pd.DataFrame([
            {"Organ":o,"Risk (%)":r[0],"Risk Color":r[1],"Explanation":r[2]}
            for o,r in sorted(organ_risks.items(),key=lambda x:x[1][0],reverse=True)
        ])
        st.download_button("🫀 Organ Risk CSV",
                           data=organ_df.to_csv(index=False),
                           file_name=f"OrganRisk_{patient_name}.csv",
                           mime="text/csv", use_container_width=True)


# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class='footer'>
  ExAI-PanC · TCGA-PAAD · GTEx Pancreas · Clinical Validation Cohorts<br>
  AUC 0.999 · C-index 0.857 · Research prototype only
</div>
""", unsafe_allow_html=True)
