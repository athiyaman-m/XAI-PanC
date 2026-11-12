import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

BG = "#0f172a"
PB = "#1e293b"
GC = "#334155"
FC = "#e2e8f0"
FF = "Inter, sans-serif"


def _lay(title="", h=280):
    return dict(paper_bgcolor=BG, plot_bgcolor=PB, font=dict(family=FF, color=FC),
                title=dict(text=title, font=dict(color="#f8fafc", size=13)),
                margin=dict(l=50, r=30, t=40, b=40), height=h)


def gauge_chart(prob: float) -> go.Figure:
    col = "#f87171" if prob > .5 else "#fbbf24" if prob > .3 else "#34d399"
    fig = go.Figure(go.Indicator(mode="gauge+number", value=prob * 100,
        number=dict(suffix="%", font=dict(size=38, color="#f8fafc", family=FF)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=GC, tickfont=dict(color="#cbd5e1", size=11)),
                   bar=dict(color=col, thickness=.2), bgcolor=BG, bordercolor=GC, borderwidth=1,
                   steps=[dict(range=[0, 30], color="rgba(52,211,153,.10)"),
                          dict(range=[30, 60], color="rgba(251,191,36,.10)"),
                          dict(range=[60, 100], color="rgba(248,113,113,.10)")],
                   threshold=dict(line=dict(color="#60a5fa", width=3), value=prob * 100)),
        title=dict(text="Cancer Probability", font=dict(size=14, color="#cbd5e1", family=FF))))
    fig.update_layout(**_lay(h=240))
    return fig


def shap_chart(shap_vals: dict) -> go.Figure:
    items = sorted(shap_vals.items(), key=lambda x: abs(x[1]))
    feats, vals = [i[0] for i in items], [i[1] for i in items]
    colors = ["#f87171" if v > 0 else "#60a5fa" for v in vals]
    fig = go.Figure(go.Bar(x=vals, y=feats, orientation="h", marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v:+.3f}" for v in vals], textposition="outside",
        textfont=dict(color="#e2e8f0", size=10, family="JetBrains Mono"),
        hovertemplate="<b>%{y}</b><br>SHAP: %{x:+.4f}<extra></extra>"))
    fig.add_vline(x=0, line_color=GC, line_width=1.5)
    l = _lay("Feature Attribution (SHAP)", 340)
    l["xaxis"] = dict(title="SHAP Value", color="#cbd5e1", gridcolor=GC, zeroline=False)
    l["yaxis"] = dict(color="#cbd5e1", tickfont=dict(size=10))
    l["margin"]["l"] = 140
    fig.update_layout(**l)
    return fig


def tx_chart(resp: dict) -> go.Figure:
    txs, probs = list(resp.keys()), [v * 100 for v in resp.values()]
    colors = ["#34d399" if p > 60 else "#fbbf24" if p > 40 else "#f87171" for p in probs]
    fig = go.Figure(go.Bar(x=txs, y=probs, marker=dict(color=colors, opacity=.85, line=dict(width=0)),
        text=[f"{p:.0f}%" for p in probs], textposition="outside",
        textfont=dict(color="#f8fafc", size=11)))
    fig.add_hline(y=50, line_dash="dot", line_color="#64748b",
                  annotation_text="50% threshold", annotation_font_color="#cbd5e1")
    l = _lay("Treatment Response", 280)
    l["yaxis"] = dict(title="Response Probability (%)", range=[0, 108], color="#cbd5e1", gridcolor=GC)
    l["xaxis"] = dict(color="#cbd5e1", tickfont=dict(size=10))
    fig.update_layout(**l)
    return fig


def km_chart(surv: dict) -> go.Figure:
    t = np.linspace(0, 36, 120)
    wb = lambda lam, t, k=1.15: np.exp(-(lam * t) ** k)
    lam = np.log(2) / max(surv["median_months"], 1)
    fig = go.Figure()
    for lbl, (l_, col, dash) in [("Low Risk", (0.018, "#34d399", "dot")),
                                  ("Medium Risk", (0.048, "#fbbf24", "dot")),
                                  ("High Risk", (0.120, "#f87171", "dot"))]:
        fig.add_trace(go.Scatter(x=t, y=wb(l_, t) * 100, mode="lines", name=lbl,
                                 line=dict(color=col, width=1.5, dash=dash), opacity=.5))
    fig.add_trace(go.Scatter(x=t, y=wb(lam, t) * 100, mode="lines",
        name=f"<b>This Patient ({surv['risk_group']})</b>",
        line=dict(color=surv["risk_color"], width=3)))
    fig.add_vline(x=surv["median_months"], line_dash="dash", line_color="#93c5fd", line_width=1.5,
                  annotation_text=f"Median: {surv['median_months']}m",
                  annotation_font_color="#93c5fd", annotation_font_size=10)
    l = _lay("Projected Survival Curve", 320)
    l["xaxis"] = dict(title="Time (Months)", color="#cbd5e1", gridcolor=GC)
    l["yaxis"] = dict(title="Survival Probability (%)", range=[0, 105], color="#cbd5e1", gridcolor=GC)
    l["legend"] = dict(bgcolor="rgba(15,23,42,.9)", bordercolor=GC, borderwidth=1, font=dict(size=10))
    fig.update_layout(**l)
    return fig


def plot_organ_radar(organ_risks: dict) -> go.Figure:
    organs = list(organ_risks.keys())
    values = [organ_risks[o][0] for o in organs]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values + [values[0]], theta=organs + [organs[0]],
        fill="toself", fillcolor="rgba(248,113,113,.15)", line=dict(color="#f87171", width=2),
        name="Patient Risk", hovertemplate="<b>%{theta}</b><br>Risk: %{r}%<extra></extra>"))
    fig.add_trace(go.Scatterpolar(r=[5] * len(organs) + [5], theta=organs + [organs[0]],
        fill="toself", fillcolor="rgba(52,211,153,.08)", line=dict(color="#34d399", width=1.5, dash="dot"),
        name="Normal Ref."))
    fig.update_layout(paper_bgcolor="#0f172a",
        polar=dict(bgcolor="#1e293b", radialaxis=dict(visible=True, range=[0, 100], gridcolor="#334155",
                   tickcolor="#93c5fd", tickfont=dict(size=9)),
                   angularaxis=dict(gridcolor="#334155", tickcolor="#93c5fd", tickfont=dict(size=10))),
        legend=dict(bgcolor="rgba(15,23,42,.9)", bordercolor="#334155", borderwidth=1, font=dict(size=10)),
        margin=dict(l=30, r=30, t=10, b=10), height=320, showlegend=False)
    return fig


def plot_biomarker_timeline(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=2, cols=2, subplot_titles=["CA 19-9", "CEA", "Bilirubin", "Amylase"])
    for col_name, color, thresh, row, col in [
        ("CA 19-9 (U/mL)", "#f87171", 37, 1, 1),
        ("CEA (ng/mL)", "#fbbf24", 5, 1, 2),
        ("Bilirubin (mg/dL)", "#60a5fa", 1.2, 2, 1),
        ("Amylase (U/L)", "#a78bfa", 110, 2, 2),
    ]:
        fig.add_trace(go.Scatter(x=df["Month"], y=df[col_name], name=col_name,
            mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=5, color=color)),
            row=row, col=col)
        fig.add_hline(y=thresh, line_dash="dot", line_color=color, line_width=1, opacity=.5, row=row, col=col)
    fig.update_layout(paper_bgcolor="#0f172a", plot_bgcolor="#1e293b", font=dict(family="Inter", color="#e2e8f0"),
                      margin=dict(l=30, r=10, t=20, b=10), height=260, showlegend=False)
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(gridcolor="#334155", row=i, col=j)
            fig.update_yaxes(gridcolor="#334155", row=i, col=j)
    return fig
