"""Plotly-Figuren (alle Achsen fest, `fixedrange`, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import numpy as np
import plotly.graph_objects as go

import clg_constants as C

_LAYOUT = dict(margin=dict(l=10, r=10, t=40, b=40), height=340, legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0))


def _fest(fig: go.Figure) -> go.Figure:
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _farbe(key: str) -> str:
    return {"naiv": C.COLOR_NAIV, "cr1z": C.COLOR_ROBUST, "cr1t": C.COLOR_ROBUST, "mlz": C.COLOR_MIXED, "rez": C.COLOR_MIXED, "ret": C.COLOR_MIXED, "kmm": C.COLOR_MEAN,
            "fe": C.COLOR_MEAN}.get(key, C.COLOR_NEUTRAL)


def build_forest(intervals: list[dict], wahr: float, titel: str) -> go.Figure:
    """Die 95-%-Intervalle aller Verfahren aus einer Historie; die senkrechte Linie ist der wahre Wert. Intervalle, die ihn verfehlen, sind gestrichelt."""
    fig = go.Figure()
    for i, iv in enumerate(intervals):
        fig.add_trace(go.Scatter(x=[iv["lo"], iv["hi"]], y=[iv["label"]] * 2, mode="lines", showlegend=False, hoverinfo="skip",
                                 line=dict(color=_farbe(iv["key"]), width=5, dash="solid" if iv["trifft"] else "dot")))
        fig.add_trace(go.Scatter(x=[iv["est"]], y=[iv["label"]], mode="markers", showlegend=False, marker=dict(color=_farbe(iv["key"]), size=9),
                                 hovertemplate=f"{iv['label']}: {iv['est']:.3f} ({iv['lo']:.3f} bis {iv['hi']:.3f})<extra></extra>"))
    fig.add_vline(x=wahr, line=dict(color=C.COLOR_TRUE, dash="dash"), annotation_text="wahrer Wert", annotation_position="top")
    fig.update_layout(title=titel, xaxis_title="Koeffizient (log-Skala der Verweilzeit)", **{**_LAYOUT, "height": 330})
    fig.update_yaxes(autorange="reversed")
    return _fest(fig)


def build_criterion(rho: np.ndarray, reml: np.ndarray, ml: np.ndarray, rho_reml: float, rho_ml: float) -> go.Figure:
    """Profiliertes Kriterium über der Intraklassenkorrelation ρ = λ/(1 + λ) (relativ zum Minimum auf dem Gitter): das Minimum ist die Schätzung, am linken Rand ρ = 0 die Randlösung."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rho, y=reml, mode="lines", name="REML", line=dict(color=C.COLOR_MIXED)))
    fig.add_trace(go.Scatter(x=rho, y=ml, mode="lines", name="ML", line=dict(color=C.COLOR_ROBUST, dash="dash")))
    for x, nm, col in ((rho_reml, "REML-Schätzung", C.COLOR_MIXED), (rho_ml, "ML-Schätzung", C.COLOR_ROBUST)):
        fig.add_vline(x=x, line=dict(color=col, dash="dot", width=1))
    fig.update_layout(title="Profiliertes Kriterium über der Intraklassenkorrelation ρ", xaxis_title="ρ = σ_u²/(σ_u² + σ²)", yaxis_title="Kriterium − Minimum (kleiner ist besser)",
                      **_LAYOUT)
    fig.update_xaxes(range=[0, 1])
    return _fest(fig)


def build_effective(n: int, wirksam: float, titel: str) -> go.Figure:
    """Zeilenzahl gegen wirksame Zahl unabhängiger Container (N geteilt durch Designeffekt)."""
    fig = go.Figure(go.Bar(x=[n, wirksam], y=["Container in der Tabelle", "wirksam (N / Designeffekt)"], orientation="h",
                           marker_color=[C.COLOR_NEUTRAL, C.COLOR_NAIV], text=[f"{n}", f"{wirksam:.0f}"], textposition="outside"))
    fig.update_layout(title=titel, xaxis_range=[0, max(n, wirksam) * 1.15], **{**_LAYOUT, "height": 220})
    fig.update_yaxes(autorange="reversed")
    return _fest(fig)


def build_coverage(tabellen: dict[str, list[dict]], titel: str) -> go.Figure:
    """Überdeckung je Verfahren als Balken mit ±2 Standardfehlern, eine Balkenreihe je Quelle (vorgerechnet, live); die Linie ist der Sollwert 95 %."""
    fig = go.Figure()
    farben = [C.COLOR_MIXED, C.COLOR_MEAN]
    for i, (quelle, tab) in enumerate(tabellen.items()):
        fig.add_trace(go.Bar(x=[t["label"] for t in tab], y=[100 * t["share"] for t in tab], name=quelle, marker_color=farben[i % 2],
                             error_y=dict(type="data", array=[200 * t["se"] for t in tab], visible=True),
                             text=[f"{100 * t['share']:.1f}".replace(".", ",") for t in tab], textposition="outside"))
    fig.add_hline(y=95, line=dict(color=C.COLOR_TRUE, dash="dash"), annotation_text="Soll 95 %", annotation_position="top left")
    fig.update_layout(title=titel, yaxis_title="Überdeckung (%)", yaxis_range=[0, 112], barmode="group", **_LAYOUT)
    return _fest(fig)


def build_rand(row: dict, formel: float | None) -> go.Figure:
    """Anteil der Historien mit σ̂_u² = 0 bei ML und REML (aus der Messreihe) und die geschlossene Formel des ausgewogenen Falls (REML) als Marke."""
    n = row["ok"]
    werte = [100 * row["rand_ml"] / n, 100 * row["rand_re"] / n]
    fig = go.Figure(go.Bar(x=["ML", "REML"], y=werte, marker_color=[C.COLOR_ROBUST, C.COLOR_MIXED], text=[f"{v:.1f}".replace(".", ",") + " %" for v in werte], textposition="outside"))
    if formel is not None:
        fig.add_hline(y=100 * formel, line=dict(color=C.COLOR_TRUE, dash="dash"), annotation_text="Formel (ausgewogen, REML)", annotation_position="top left")
    fig.update_layout(title="Anteil der Historien am Rand (σ̂_u² = 0)", yaxis_title="Historien (%)", yaxis_range=[0, max(werte + [100 * (formel or 0)]) * 1.3 + 3], **{**_LAYOUT, "height": 300})
    return _fest(fig)


def build_customers(uk, nominal: float) -> go.Figure:
    """Die zwölf Kundeneffekte des Generators gegen das Band ±nominale Streuung."""
    uk = np.asarray(uk)
    fig = go.Figure()
    fig.add_hrect(y0=-nominal, y1=nominal, fillcolor="rgba(150,150,150,0.2)", line_width=0, annotation_text="± nominale Streuung 0,30", annotation_position="top left")
    fig.add_trace(go.Scatter(x=[f"K{i + 1}" for i in range(len(uk))], y=uk, mode="markers", marker=dict(color=C.COLOR_MIXED, size=10), showlegend=False,
                             hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(title="Die zwölf Kundeneffekte des Generators (log-Skala)", yaxis_title="u_k", **{**_LAYOUT, "height": 300})
    return _fest(fig)


def build_su2(werte, wahr: float) -> go.Figure:
    """Verteilung von σ̂_u² (REML) über die Historien gegen den wahren Wert σ_u²."""
    fig = go.Figure(go.Histogram(x=werte, nbinsx=30, marker_color=C.COLOR_MIXED, showlegend=False))
    fig.add_vline(x=wahr, line=dict(color=C.COLOR_TRUE, dash="dash"), annotation_text="wahr", annotation_position="top")
    fig.update_layout(title="Schätzungen σ̂_u² (REML) über die Historien", xaxis_title="σ̂_u²", yaxis_title="Historien", **{**_LAYOUT, "height": 300})
    return _fest(fig)


def build_su2_sd(grid: list[dict]) -> go.Figure:
    """Streuung von σ̂_u² gegen Container je Kunde, eine Linie je Kundenzahl (vorgerechnet), dazu die Grenze σ_u²·√(2/(m−1)) für sehr viele Container."""
    fig = go.Figure()
    farben = {6: C.COLOR_NAIV, 12: C.COLOR_MIXED, 48: C.COLOR_MEAN}
    for m in C.SU2_GRID_M:
        pts = [g for g in grid if g["m"] == m]
        if not pts:
            continue
        fig.add_trace(go.Scatter(x=[g["k"] for g in pts], y=[g["sd"] for g in pts], mode="lines+markers", name=f"m = {m}", line=dict(color=farben[m])))
        fig.add_hline(y=pts[0]["grenze"], line=dict(color=farben[m], dash="dot", width=1))
    fig.update_layout(title="Streuung von σ̂_u²: mehr Kunden helfen, mehr Container kaum", xaxis_title="Container je Kunde (im Mittel)", yaxis_title="Streuung von σ̂_u²",
                      xaxis_type="log", **_LAYOUT)
    return _fest(fig)
