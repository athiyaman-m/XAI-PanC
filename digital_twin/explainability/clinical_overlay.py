from ..anatomy.organ_models import Organ


def _status_colour(score: float, high_is_bad: bool = True) -> str:
    if high_is_bad:
        if score >= 70: return "#f87171"
        if score >= 45: return "#fbbf24"
        if score >= 20: return "#fde68a"
        return "#34d399"
    else:
        if score >= 70: return "#34d399"
        if score >= 45: return "#fde68a"
        if score >= 20: return "#fbbf24"
        return "#f87171"


def organ_hover_html(organ: Organ) -> str:
    health_col = _status_colour(organ.health_score, high_is_bad=False)
    risk_col = _status_colour(organ.risk_score, high_is_bad=True)
    bio_rows = ""
    for k, v in sorted(organ.biomarkers.items()):
        bio_rows += (
            f'<tr><td style="color:#94a3b8;padding-right:12px;">{k}</td>'
            f'<td style="color:#f1f5f9;font-weight:600;">{v:.1f}</td></tr>'
        )
    shap_rows = ""
    for k, v in sorted(organ.shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
        sc = "#f87171" if v > 0 else "#60a5fa"
        shap_rows += (
            f'<tr><td style="color:#94a3b8;padding-right:12px;">{k}</td>'
            f'<td style="color:{sc};font-weight:600;">{v:+.4f}</td></tr>'
        )
    html = (
        '<div style="font-family:Inter,sans-serif;background:#0f172a;border:1px solid #334155;'
        'border-radius:10px;padding:14px 18px;min-width:220px;">'
        '<div style="font-size:15px;font-weight:700;color:#f8fafc;margin-bottom:8px;'
        'border-bottom:1px solid #334155;padding-bottom:6px;">'
        f'{organ.name}'
        '</div>'
        '<table style="font-size:13px;line-height:1.9;">'
        f'<tr><td style="color:#94a3b8;">Health</td>'
        f'<td style="color:{health_col};font-weight:700;">{organ.health_score:.0f}%</td></tr>'
        f'<tr><td style="color:#94a3b8;">Risk</td>'
        f'<td style="color:{risk_col};font-weight:700;">{organ.risk_score:.0f}%</td></tr>'
        f'<tr><td style="color:#94a3b8;">Stage</td>'
        f'<td style="color:#93c5fd;font-weight:600;">{organ.disease_stage}</td></tr>'
        f'<tr><td style="color:#94a3b8;">Status</td>'
        f'<td style="color:#f1f5f9;">{organ.status_text()}</td></tr>'
        f'<tr><td style="color:#94a3b8;">Lesions</td>'
        f'<td style="color:#f1f5f9;">{len(organ.lesions)}</td></tr>'
        '</table>'
    )
    if bio_rows:
        html += ('<div style="margin-top:8px;font-size:12px;color:#93c5fd;font-weight:600;border-top:1px solid #334155;'
                 'padding-top:6px;">Biomarkers</div>'
                 f'<table style="font-size:12px;line-height:1.7;">{bio_rows}</table>')
    if shap_rows:
        html += ('<div style="margin-top:8px;font-size:12px;color:#93c5fd;font-weight:600;border-top:1px solid #334155;'
                 'padding-top:6px;">SHAP Contributions</div>'
                 f'<table style="font-size:12px;line-height:1.7;">{shap_rows}</table>')
    html += '</div>'
    return html


def patient_panel_html(engine) -> str:
    pdata = engine.patient_data
    overall_health = engine.overall_health
    overall_risk = engine.overall_risk
    health_col = _status_colour(overall_health, high_is_bad=False)
    risk_col = _status_colour(overall_risk, high_is_bad=True)
    top_organ = None
    if engine.organs:
        top_organ = max(engine.organs.values(), key=lambda o: o.risk_score)
    organ_bars = ""
    for name in sorted(engine.organs):
        o = engine.organs[name]
        bc = o.colour
        hw = min(100, o.health_score)
        organ_bars += (
            '<div style="margin-bottom:6px;">'
            '<div style="display:flex;justify-content:space-between;font-size:12px;">'
            f'<span style="color:#94a3b8;">{name}</span>'
            f'<span style="color:{bc};font-weight:700;">{o.health_score:.0f}% / {o.risk_score:.0f}%</span>'
            '</div>'
            '<div style="background:#334155;border-radius:3px;height:5px;position:relative;">'
            f'<div style="position:absolute;top:0;left:0;height:100%;width:{hw}%;'
            f'background:linear-gradient(90deg,{bc},{bc}88);border-radius:3px;"></div>'
            '</div></div>'
        )
    top_section = ""
    if top_organ:
        tc = top_organ.colour
        top_section = (
            '<div style="margin-top:12px;border-top:1px solid #334155;padding-top:10px;">'
            '<div style="font-size:12px;color:#93c5fd;font-weight:700;margin-bottom:5px;">HIGHEST RISK</div>'
            f'<div style="font-size:22px;font-weight:800;color:{tc};">{top_organ.name}</div>'
            f'<div style="font-size:12px;color:#94a3b8;">{top_organ.risk_score:.0f}% risk - {top_organ.disease_stage}</div>'
            '</div>'
        )
    html = (
        '<div style="font-family:Inter,sans-serif;background:linear-gradient(145deg,#1e293b,#0f172a);'
        'border:1px solid #334155;border-radius:12px;padding:18px;">'
        '<div style="font-size:14px;font-weight:700;color:#60a5fa;margin-bottom:12px;'
        'text-transform:uppercase;letter-spacing:1px;border-bottom:1px solid #334155;padding-bottom:10px;">'
        'PATIENT DIGITAL TWIN</div>'
        '<table style="font-size:13px;line-height:2.2;width:100%;">'
        f'<tr><td style="color:#94a3b8;">Age</td><td style="color:#f8fafc;font-weight:600;">{pdata.get("age", "-")}</td></tr>'
        f'<tr><td style="color:#94a3b8;">Sex</td><td style="color:#f8fafc;font-weight:600;">{pdata.get("sex", "-")}</td></tr>'
        f'<tr><td style="color:#94a3b8;">Overall Health</td><td style="color:{health_col};font-weight:700;">{overall_health:.0f}%</td></tr>'
        f'<tr><td style="color:#94a3b8;">Overall Risk</td><td style="color:{risk_col};font-weight:700;">{overall_risk:.0f}%</td></tr>'
        f'<tr><td style="color:#94a3b8;">Current Stage</td><td style="color:#93c5fd;font-weight:600;">{engine.predictions.get("stage_label", "-")}</td></tr>'
        '</table>'
        f'<div style="margin-top:12px;border-top:1px solid #334155;padding-top:10px;">'
        f'<div style="font-size:12px;color:#93c5fd;font-weight:700;margin-bottom:8px;">ORGAN BREAKDOWN</div>'
        f'{organ_bars}</div>'
        f'{top_section}'
        '<div style="margin-top:12px;border-top:1px solid #334155;padding-top:10px;font-size:11px;color:#64748b;text-align:center;">'
        'Digital Twin v3.0 - AI-powered - Research use only</div>'
        '</div>'
    )
    return html
