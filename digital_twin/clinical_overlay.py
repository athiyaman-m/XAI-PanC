"""
clinical_overlay.py — Hover overlays and the Patient Digital Twin panel.

Provides rich HTML tooltip content and a side-panel summarising
the patient's digital twin state. All HTML is plain string-safe
(no nested f-string conditional complexity).
"""

from .organ_models import Organ


def _status_colour(score: float, high_is_bad: bool = True) -> str:
    if high_is_bad:
        if score >= 70: return "#e74c3c"
        if score >= 45: return "#f39c12"
        if score >= 20: return "#f1c40f"
        return "#2ecc71"
    else:
        if score >= 70: return "#2ecc71"
        if score >= 45: return "#f1c40f"
        if score >= 20: return "#f39c12"
        return "#e74c3c"


def organ_hover_html(organ: Organ) -> str:
    """Rich HTML tooltip for an organ when hovered."""
    health_col = _status_colour(organ.health_score, high_is_bad=False)
    risk_col = _status_colour(organ.risk_score, high_is_bad=True)

    # Build biomarker rows
    bio_rows = ""
    for k, v in sorted(organ.biomarkers.items()):
        bio_rows += (
            f'<tr><td style="color:#7a9fc5;padding-right:12px;">{k}</td>'
            f'<td style="color:#e0e6f0;font-weight:600;">{v:.1f}</td></tr>'
        )

    # Build SHAP rows
    shap_rows = ""
    for k, v in sorted(organ.shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        sc = "#e74c3c" if v > 0 else "#3b82f6"
        shap_rows += (
            f'<tr><td style="color:#7a9fc5;padding-right:12px;">{k}</td>'
            f'<td style="color:{sc};font-weight:600;">{v:+.4f}</td></tr>'
        )

    html = (
        '<div style="font-family:sans-serif;background:#0b1222;border:1px solid #1a3555;'
        'border-radius:10px;padding:12px 16px;min-width:220px;">'
        '<div style="font-size:14px;font-weight:700;color:#fff;margin-bottom:6px;'
        'border-bottom:1px solid #1a3555;padding-bottom:4px;">'
        f'{organ.name}'
        '</div>'
        '<table style="font-size:12px;line-height:1.8;">'
        f'<tr><td style="color:#7a9fc5;">Health</td>'
        f'<td style="color:{health_col};font-weight:700;">{organ.health_score:.0f}%</td></tr>'
        f'<tr><td style="color:#7a9fc5;">Risk</td>'
        f'<td style="color:{risk_col};font-weight:700;">{organ.risk_score:.0f}%</td></tr>'
        f'<tr><td style="color:#7a9fc5;">Stage</td>'
        f'<td style="color:#60a5fa;font-weight:600;">{organ.disease_stage}</td></tr>'
        f'<tr><td style="color:#7a9fc5;">Status</td>'
        f'<td style="color:#e0e6f0;">{organ.status_text()}</td></tr>'
        f'<tr><td style="color:#7a9fc5;">Lesions</td>'
        f'<td style="color:#e0e6f0;">{len(organ.lesions)}</td></tr>'
        '</table>'
    )

    if bio_rows:
        html += (
            '<div style="margin-top:6px;font-size:11px;color:#8bb8e0;border-top:1px solid #1a3555;'
            'padding-top:4px;"><b>Biomarkers</b></div>'
            f'<table style="font-size:11px;line-height:1.6;">{bio_rows}</table>'
        )

    if shap_rows:
        html += (
            '<div style="margin-top:6px;font-size:11px;color:#8bb8e0;border-top:1px solid #1a3555;'
            'padding-top:4px;"><b>SHAP Contributions</b></div>'
            f'<table style="font-size:11px;line-height:1.6;">{shap_rows}</table>'
        )

    if organ.explanation:
        html += (
            '<div style="margin-top:6px;font-size:11px;color:#7a9fc5;border-top:1px solid #1a3555;'
            f'padding-top:4px;">{organ.explanation}</div>'
        )

    if organ.recommendation:
        html += (
            '<div style="margin-top:4px;font-size:11px;color:#2ecc71;font-weight:600;">'
            f'{organ.recommendation}</div>'
        )

    html += '</div>'
    return html


def patient_panel_html(engine) -> str:
    """
    Render the Patient Digital Twin side panel as HTML.
    """
    pdata = engine.patient_data
    overall_health = engine.overall_health
    overall_risk = engine.overall_risk
    health_col = _status_colour(overall_health, high_is_bad=False)
    risk_col = _status_colour(overall_risk, high_is_bad=True)

    top_organ = None
    if engine.organs:
        top_organ = max(engine.organs.values(), key=lambda o: o.risk_score)

    # Organ breakdown bars
    organ_bars = ""
    for name in sorted(engine.organs):
        o = engine.organs[name]
        bc = o.colour
        hw = min(100, o.health_score)
        organ_bars += (
            '<div style="margin-bottom:5px;">'
            '<div style="display:flex;justify-content:space-between;font-size:11px;">'
            f'<span style="color:#7a9fc5;">{name}</span>'
            f'<span style="color:{bc};font-weight:600;">{o.health_score:.0f}% / {o.risk_score:.0f}%</span>'
            '</div>'
            '<div style="background:#1a3555;border-radius:3px;height:4px;position:relative;">'
            f'<div style="position:absolute;top:0;left:0;height:100%;width:{hw}%;'
            f'background:linear-gradient(90deg,{bc},{bc}88);border-radius:3px;"></div>'
            '</div>'
            '</div>'
        )

    top_section = ""
    if top_organ:
        tc = top_organ.colour
        top_section = (
            '<div style="margin-top:10px;border-top:1px solid #1a3555;padding-top:8px;">'
            '<div style="font-size:11px;color:#8bb8e0;font-weight:600;margin-bottom:4px;">HIGHEST RISK</div>'
            f'<div style="font-size:20px;font-weight:800;color:{tc};">{top_organ.name}</div>'
            f'<div style="font-size:11px;color:#7a9fc5;">{top_organ.risk_score:.0f}% risk - {top_organ.disease_stage}</div>'
            '</div>'
        )

    html = (
        '<div style="font-family:sans-serif;background:linear-gradient(145deg,#0e1e30,#0b1828);'
        'border:1px solid #1a3555;border-radius:12px;padding:16px;">'

        '<div style="font-size:13px;font-weight:700;color:#60a5fa;margin-bottom:10px;'
        'text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid #1a3555;padding-bottom:8px;">'
        'PATIENT DIGITAL TWIN</div>'

        '<table style="font-size:12px;line-height:2;width:100%;">'
        f'<tr><td style="color:#7a9fc5;">Age</td><td style="color:#fff;font-weight:600;">{pdata.get("age", "-")}</td></tr>'
        f'<tr><td style="color:#7a9fc5;">Sex</td><td style="color:#fff;font-weight:600;">{pdata.get("sex", "-")}</td></tr>'
        f'<tr><td style="color:#7a9fc5;">Overall Health</td><td style="color:{health_col};font-weight:700;">{overall_health:.0f}%</td></tr>'
        f'<tr><td style="color:#7a9fc5;">Overall Risk</td><td style="color:{risk_col};font-weight:700;">{overall_risk:.0f}%</td></tr>'
        f'<tr><td style="color:#7a9fc5;">Current Stage</td><td style="color:#60a5fa;font-weight:600;">{engine.predictions.get("stage_label", "-")}</td></tr>'
        '</table>'

        '<div style="margin-top:10px;border-top:1px solid #1a3555;padding-top:8px;">'
        '<div style="font-size:11px;color:#8bb8e0;font-weight:600;margin-bottom:6px;">ORGAN BREAKDOWN</div>'
        f'{organ_bars}'
        '</div>'

        f'{top_section}'

        '<div style="margin-top:10px;border-top:1px solid #1a3555;padding-top:8px;font-size:10px;color:#1a3a5a;text-align:center;">'
        'Digital Twin v2.0 - AI-powered - Research use only</div>'

        '</div>'
    )
    return html
