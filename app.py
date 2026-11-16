"""Cluster und gemischte Modelle – interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Mehrere Container desselben Kunden sind keine unabhängigen Beobachtungen: für ein Merkmal, das nur der Kunde hat, zählt die Kundenzahl, nicht die Zeilenzahl.
Vehikel: Import-Container der Verweilzeit-Welt, log-Verweilzeit, Kunden als Cluster. Alle Formeln und Befunde sind eigene Nachrechnung in diesem Modell, bei diesen
Parametern (Zufallsachsenabschnitt, normale Fehler, konstante Streuung, 6 bis 48 Kunden); Kern: clg_mixed.py, Vehikel: clg_vehicle.py.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import clg_constants as C
import clg_evaluation as ev
import clg_mixed as MX
import clg_vehicle as V
from clg_evaluation import Settings, de, de_pct
from clg_layout import dunkles_design, kopf_css
from clg_presets import (
    SETTING_SPECS,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
    value_of,
)
from clg_visualization import (
    build_coverage,
    build_criterion,
    build_customers,
    build_effective,
    build_forest,
    build_rand,
    build_su2,
    build_su2_sd,
)

st.set_page_config(page_title="Cluster und gemischte Modelle – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _hand(text):
    return ev.hand_analysis(ev.parse_hand(text))


@st.cache_data(show_spinner=False)
def _history(m, k, su, art, seed):
    h = V.draw_history(seed, m, k, su, art)
    return dict(rv=V.history_intervals(h, 1), zoll=V.history_intervals(h, 2), n=len(h.y), m=h.m)


@st.cache_data(show_spinner=False)
def _design(s):
    return ev.analyse_design(s)


@st.cache_data(show_spinner="Simuliere die Historien …")
def _live_row(m, k, su, art, reps, seed):
    return V.simulate(m, k, su, reps, seed, art)


@st.cache_data(show_spinner="Schätze σ̂_u² über die Historien …")
def _su2_live(m, k, su, art, reps, seed, mim):
    return V.su2_series(m, k, su, reps, seed, art, mim)


@st.cache_data(show_spinner=False)
def _generator():
    return ev.generator_summary()


@st.cache_data(show_spinner=False)
def _rang(m, k, seed):
    """Rang der Spalten [Kunden-Dummies, Rahmenvertrag] an einer Historie: der Wechsel zu festen Kunden-Effekten geht für ein Kundenmerkmal nicht."""
    h = V.draw_history(seed, m, k, 0.3, "half")
    D = np.zeros((len(h.y), h.m))
    D[np.arange(len(h.y)), h.g] = 1.0
    X = np.column_stack([D, h.X[:, 1]])
    return int(np.linalg.matrix_rank(X)), X.shape[1]


def current_settings() -> Settings:
    v = value_of
    return Settings(m=int(v("m_slider")), k=int(v("k_slider")), su=float(v("su_slider")), art=v("art_select"), seed=int(v("seed_input")), hand=v("hand_text"),
                    merk=v("merk_select"), reps=int(v("reps_select")), mim=bool(v("mim_toggle")))


def _lade_hand(name):
    if name == "Handbeispiel":
        st.session_state["hand_text"] = C.HAND_BEISPIEL
    elif name == "Randfall":
        st.session_state["hand_text"] = C.HAND_RANDFALL
    else:
        st.session_state["hand_text"] = ev.random_hand(int(st.session_state.get("seed_input", C.DEFAULT_SEED)))


def _zeit_schaetzung(m, k, reps) -> float:
    """Grobe Rechenzeit der Live-Simulation in Sekunden (gemessen: etwa 1,4 ms + 0,4 µs je Container und Historie)."""
    return reps * (1.4 + 0.0004 * m * k) / 1000.0


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in clg_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧩 Cluster und gemischte Modelle")
st.markdown(
    """
**Mehrere Container desselben Kunden sind keine unabhängigen Beobachtungen.** Gehört ein Merkmal nur dem **Kunden** (hier ein erfundener **Rahmenvertrag**), zählt die **Kundenzahl**, nicht
die Zeilenzahl: das naive Intervall einer Regression ist dann viel zu schmal. Die Demo misst das an den **Import-Containern** der Verweilzeit-Welt (log-Verweilzeit, Kunden als Cluster)
und vergleicht das naive Intervall mit **cluster-robusten Fehlern (CR1)**, dem **gemischten Modell (REML)** und der **Kunden-Mittel-Methode**. Sie zeigt auch, wann das gemischte
Modell selbst am **Rand** σ̂_u² = 0 liegt und warum ein Merkmal **innerhalb** der Kunden anders ist. Alle Formeln und Befunde sind **eigene Nachrechnung in diesem Modell**
(Zufallsachsenabschnitt, normale Fehler, konstante Streuung, 6 bis 48 Kunden); Literatur dazu steht in der README mit dem Grad ihrer Einsicht.

Weiter unten: das **Ergebnis** einer Historie, die **Kernfragen** mit Verdikt, fünf Experimente (🔬), der Abschnitt **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert der Vergleich", expanded=True):
    st.markdown(
        """
1. **Zeilen und Kunden:** N Import-Container gehören zu m Kunden, im Mittel k̄ = N/m je Kunde; der Kunde jedes Containers wird zufällig gezogen, die Kunden sind also **ungleich groß**. Die Antwort ist
   die **log-Verweilzeit**: ln 3,5 + 0,7·Zollhalt + 0,3·Wochenende + β_R·Rahmenvertrag + u_Kunde + 0,55·ε, mit u_Kunde ~ N(0, σ_u²) und ε ~ N(0, 1).
   Der **Rahmenvertrag** (β_R = −0,30) ist **erfunden** und gilt für ganze Kunden; Zollhalt und Wochenende wechseln **innerhalb** der Kunden.
2. **Eine Historie** ist eine ganze Tabelle aus m Kunden mit frisch gezogenen Kundeneffekten; sie ist die **Zähleinheit** jeder Überdeckung (Anteil der Historien, in denen das 95-%-Intervall den wahren Wert trifft).
3. **Verfahren:** *naiv* (gewöhnliche Kleinste Quadrate, t(N − p)), *CR1* (cluster-robuste Varianz), *ML* und *REML* (gemischtes Modell mit einem Zufallsachsenabschnitt je Kunde), *Kunden-Mittel-Methode*
   (Regression der Kundenmittel auf das Kundenmerkmal, t(m − 2)) und *feste Kunden-Effekte* (für Merkmale innerhalb der Kunden).
4. **Bedingung:** die Aussagen gelten für diese Welt (ein Zufallsachsenabschnitt, normale Fehler, konstante Streuung, Kunden als Stichprobe einer Kundschaft); Einzelwerte aus **einer** Historie sind kein Beleg.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Import-Container und ihre Kunden**")
    st.slider("Kunden m", C.M_MIN, C.M_MAX, key="m_slider", help="Zahl der Kunden in einer Historie. Die Kundenzahl entscheidet über Kundenmerkmale.")
    st.slider("Container je Kunde (im Mittel) k̄", C.K_MIN, C.K_MAX, step=C.K_STEP, key="k_slider",
              help="Im Mittel N/m Container je Kunde; die Kundengrößen sind zufällig, also ungleich. Mehr Container ändern bei gleicher Kundenzahl wenig am Kundenmerkmal.")
    st.select_slider("Streuung zwischen den Kunden σ_u", options=C.SU_OPTIONS, key="su_slider", format_func=lambda x: de(x, 2),
                     help="Standardabweichung der Kundeneffekte auf der log-Skala. 0 heißt: es gibt keinen Kundeneffekt. Die Streuung im Kunden ist 0,55 (Import).")
    st.selectbox("Rahmenvertrag: wie ist er verteilt?", list(C.ARTEN), format_func=C.ARTEN.get, key="art_select",
                 help="0/1 bei der Hälfte der Kunden (Voreinstellung), 0/1 bei nur zwei Kunden (ungleich verteilt) oder stetig (standardnormal je Kunde, Wirkung je Standardabweichung).")
    st.markdown("**Zufall**")
    st.number_input("Zufalls-Seed", 0, C.SEED_MAX, key="seed_input", step=1,
                    help="Legt die Historie des Ergebnisses, die Live-Simulationen und den Zufallsdatensatz fest (eigener ganzzahliger Zufallsgenerator, SplitMix64).")
    st.button("🎲 Neuen Zufalls-Seed ziehen", width="stretch", on_click=randomize_seed)

s = current_settings()
sync_query_params({k: value_of(k) for k in SETTING_SPECS})
s_rv = Settings(m=s.m, k=s.k, su=s.su, art=s.art, seed=s.seed, merk=C.MERKMALE[0])

frozen_row = ev.find_frozen_row(s.m, s.k, s.su, s.art)
live_row = _live_row(s.m, s.k, s.su, s.art, ev.live_reps(s.m, s.k, s.reps), s.seed)
row, row_quelle = (frozen_row, "vorgerechnet") if frozen_row else (live_row, "live")

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: eine Historie mit Ihren Einstellungen")
hist = _history(s.m, s.k, s.su, s.art, s.seed)
design = _design(s_rv)
m1, m2, m3 = st.columns(3)
m1.metric("Container N in dieser Historie", f"{hist['n']}", delta=f"{hist['m']} Kunden mit Containern", delta_color="off")
m2.metric("Intraklassenkorrelation ρ (wahr)", de(s.rho, 3), delta="Anteil der Kundenstreuung an der Gesamtstreuung", delta_color="off")
m3.metric("Wirksame Zahl unabhängiger Container (Rahmenvertrag)", de(design["wirksam_median"], 0), delta=f"von {hist['n']} Containern", delta_color="off")
st.plotly_chart(build_forest(hist["rv"], V.BETA_R, "95-%-Intervalle für den Rahmenvertrag (eine Historie)"), width="stretch", key="main_forest")
st.caption("Eine einzige Historie mit dem Zufalls-Seed aus der Seitenleiste; gestrichelte Intervalle verfehlen den wahren Wert −0,30. Einzelne Historien sind kein Beleg; die Anteile über viele Historien stehen in den Experimenten.")
with st.expander("Dieselbe Historie für Zollhalt (Merkmal innerhalb der Kunden)"):
    st.plotly_chart(build_forest(hist["zoll"], V.BETA_Z, "95-%-Intervalle für den Zollhalt (dieselbe Historie)"), width="stretch", key="main_forest_zoll")

# --- Kernfragen ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfragen")
st.markdown("**1. Wie viele unabhängige Einheiten hat die Tabelle wirklich?**")
kind, text = ev.verdict_design(design)
getattr(st, kind)(text)
st.markdown("**2. Welches Intervall darf ich glauben?**")
kind, text = ev.verdict_coverage(row, row_quelle)
getattr(st, kind)(text)
st.markdown("**3. Wann versagt auch das gemischte Modell?**")
kind, text = ev.verdict_rand(row)
getattr(st, kind)(text)

# --- Experiment 1 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 1: Von Hand, drei Kunden")
st.markdown(
    "Kleine Tabelle, jede Zahl nachrechenbar: je Kunde eine Gruppe von Zahlen (log-Verweilzeit-Abweichungen, Einheiten beliebig), Kunden mit **|** getrennt, Zahlen mit Komma. "
    "Gleich große Kunden rechnet die App **in Brüchen** (MSB, MSW, die Schätzer von σ_u²), sonst über das profilierte Kriterium."
)
b1, b2, b3 = st.columns(3)
b1.button("Handbeispiel laden", width="stretch", on_click=_lade_hand, args=("Handbeispiel",), key="hand_btn_1")
b2.button("Randfall laden", width="stretch", on_click=_lade_hand, args=("Randfall",), key="hand_btn_2")
b3.button("Zufallsdatensatz laden (Zufalls-Seed)", width="stretch", on_click=_lade_hand, args=("Zufall",), key="hand_btn_3")
st.text_input("Zahlen je Kunde (Kunden mit | trennen, Zahlen mit Komma)", key="hand_text",
              help=f"Bis zu {C.HAND_MAX_KUNDEN} Kunden mit je bis zu {C.HAND_MAX_CONTAINER} Zahlen; ganze Zahlen, Dezimalzahlen mit Punkt oder Brüche wie 1/3.")
try:
    ha = _hand(s.hand)
except ValueError as err:
    ha = None
    st.error(f"Eingabe nicht lesbar: {err}")
if ha is not None:
    c1, c2 = st.columns([1, 1])
    with c1:
        if ha.anova is not None:
            st.markdown(f"Gesamtmittel **{ev.frac_zahl(ha.gm)}**; Kundenmittel {', '.join(ev.frac_str(c) for c in ha.kundenmittel)}; "
                        f"**MSB = {ev.frac_str(ha.anova['msb'])}**, **MSW = {ev.frac_str(ha.anova['msw'])}** (SSB = {ev.frac_str(ha.anova['ssb'])}, SSW = {ev.frac_str(ha.anova['ssw'])}).")
        else:
            st.markdown(f"Gesamtmittel **{ev.frac_zahl(ha.gm)}**; Kundenmittel {', '.join(ev.frac_zahl(c, 2) for c in ha.kundenmittel)}. "
                        "Die Kunden sind ungleich groß: MSB und MSW gibt es nicht in dieser Form, die Schätzer kommen aus dem profilierten Kriterium.")
        st.dataframe(pd.DataFrame(ev.hand_table(ha)), hide_index=True, width="stretch")
        st.markdown(
            f"Mittelwert des Datensatzes: naiver Standardfehler **{de(ha.se_naiv, 2)}** (σ̂² = Streuung aller {ha.n} Zahlen), gemischter Standardfehler (REML) **{de(ha.se_gemischt, 2)}**; "
            f"Halbbreite des 95-%-Intervalls naiv mit t({ha.n - 1}) **{de(ha.halb_naiv, 2)}**, gemischt mit t({ha.m - 1}) **{de(ha.halb_gemischt, 2)}**."
        )
    with c2:
        rho_r = ha.reml.rho if ha.reml.getrennt else 0.0
        rho_m = ha.ml.rho if ha.ml.getrennt else 0.0
        st.plotly_chart(build_criterion(ha.rho_plot, ha.krit_reml, ha.krit_ml, rho_r, rho_m), width="stretch", key="hand_krit")
    kind, text = ev.verdict_hand(ha)
    getattr(st, kind)(text)
    st.caption("Das Kriterium ist die negative Log-Likelihood (REML bzw. ML, profiliert über σ²) bis auf Konstanten, relativ zum Minimum auf dem Gitter; am linken Rand ρ = 0 liegt die Randlösung.")

# --- Experiment 2 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 2: Wie viele unabhängige Container?")
st.radio("Merkmal", C.MERKMALE, key="merk_select", horizontal=True,
         help="Rahmenvertrag gehört dem Kunden (gilt für alle seine Container), Zollhalt wechselt von Container zu Container innerhalb des Kunden.")
s = current_settings()
d2 = _design(s)
st.markdown(
    f"Der **Designeffekt** ist das Verhältnis der wahren Varianz des Koeffizienten aus einer gewöhnlichen Regression (bei der wahren Kovarianz der Tabelle, exakt über die Matrix, "
    f"{d2['anzahl']} zufällige Zuordnungen der Container zu Kunden) zur naiven Formel mit der Gesamtvarianz; N geteilt durch den Designeffekt ist die **wirksame Zahl** unabhängiger Container."
)
if d2["art"] == "rv":
    t2 = pd.DataFrame([
        dict(Größe="Container N", Wert=f"{d2['n']}"),
        dict(Größe="Designeffekt exakt (Median, 5 bis 95 %)", Wert=f"{de(d2['median'], 2)} ({de(d2['q05'], 2)} bis {de(d2['q95'], 2)})"),
        dict(Größe="Formel 1 + (k̄ − 1)ρ", Wert=de(d2["formel"], 2)),
        dict(Größe="Wirksame Zahl N/Designeffekt (Median)", Wert=de(d2["wirksam_median"], 0)),
    ])
else:
    t2 = pd.DataFrame([
        dict(Größe="Container N", Wert=f"{d2['n']}"),
        dict(Größe="wahre Varianz gegen naive Formel (Gesamtvarianz), Median", Wert=f"{de(d2['median'], 2)} ({de(d2['q05'], 2)} bis {de(d2['q95'], 2)})"),
        dict(Größe="wahre Varianz gegen die erwartete naive Varianz, Median", Wert=de(d2["median_erwartet"], 2)),
        dict(Größe="Wirksame Zahl (Median)", Wert=de(d2["wirksam_median"], 0)),
    ])
st.dataframe(t2, hide_index=True, width="stretch")
st.plotly_chart(build_effective(d2["n"], d2["wirksam_median"], "Container in der Tabelle gegen wirksame Zahl"), width="stretch", key="exp2_eff")
kind, text = ev.verdict_design(d2)
getattr(st, kind)(text)

# --- Experiment 3 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Welches Intervall deckt?")
st.select_slider("Wiederholungen der Live-Simulation (Historien)", options=C.REPS_OPTIONS, key="reps_select",
                 help="So viele Historien rechnet die App sofort neu (Zähleinheit: eine Historie). Mehr Historien verkleinern den Monte-Carlo-Fehler.")
s = current_settings()
reps_eff = ev.live_reps(s.m, s.k, s.reps)
live_row = _live_row(s.m, s.k, s.su, s.art, reps_eff, s.seed)
st.caption(f"Live-Rechnung: {live_row['ok']} Historien mit dem Zufalls-Seed aus der Seitenleiste (Rechenzeit grob {de(_zeit_schaetzung(s.m, s.k, reps_eff), 1)} s"
           + (f"; auf {reps_eff} begrenzt, damit die Rechnung nicht länger als etwa {de(ev.LIVE_BUDGET_MS / 1000, 1)} s dauert" if reps_eff < s.reps else "") + "). "
           + ("Zu dieser Einstellung gibt es auch eine **vorgerechnete** Zeile mit 1000 Historien." if frozen_row else
              "Zu dieser Einstellung gibt es keine vorgerechnete Zeile (nur Einstellungen aus der Tabelle unten sind vorgerechnet)."))
tabs = {f"live ({live_row['ok']} Historien)": ev.coverage_table(live_row, "rv")}
if frozen_row:
    tabs = {f"vorgerechnet ({frozen_row['ok']} Historien)": ev.coverage_table(frozen_row, "rv"), **tabs}
st.plotly_chart(build_coverage(tabs, "Überdeckung des 95-%-Intervalls für den Rahmenvertrag"), width="stretch", key="exp3_cov")
vergleich = []
for i, t in enumerate(ev.coverage_table(live_row, "rv")):
    zeile = {"Verfahren": t["label"], f"live ({live_row['ok']}) in %": 100 * t["share"], "± 2 Standardfehler": 200 * t["se"]}
    if frozen_row:
        tf = ev.coverage_table(frozen_row, "rv")[i]
        zeile[f"vorgerechnet ({frozen_row['ok']}) in %"] = 100 * tf["share"]
        zeile["Breite gegen naiv"] = tf["breite"]
    else:
        zeile["Breite gegen naiv"] = t["breite"]
    vergleich.append(zeile)
st.dataframe(pd.DataFrame(vergleich), hide_index=True, width="stretch",
             column_config={c: st.column_config.NumberColumn(format="%.1f") for c in pd.DataFrame(vergleich).columns if c != "Verfahren" and c != "Breite gegen naiv"}
             | {"Breite gegen naiv": st.column_config.NumberColumn(format="%.2f")})
row3, quelle3 = (frozen_row, "vorgerechnet") if frozen_row else (live_row, "live")
kind, text = ev.verdict_coverage(row3, quelle3)
getattr(st, kind)(text)
with st.expander("Zollhalt (Merkmal innerhalb der Kunden): derselbe Vergleich"):
    st.markdown("Der Zollhalt wechselt von Container zu Container; für ihn ist das naive Intervall nicht zu schmal. Die Breiten sind im Verhältnis zur naiven Breite angegeben.")
    zr = frozen_row or live_row
    st.dataframe(pd.DataFrame([{"Verfahren": t["label"], "Überdeckung in %": 100 * t["share"], "Breite gegen naiv": t["breite"]} for t in ev.coverage_table(zr, "zoll")]),
                 hide_index=True, width="stretch",
                 column_config={"Überdeckung in %": st.column_config.NumberColumn(format="%.1f"), "Breite gegen naiv": st.column_config.NumberColumn(format="%.2f")})
    p_ret = V.share(zr, "zoll", "ret")
    urteil = "über dem Sollwert 95 % (das Intervall ist zu breit)" if ev.ueber_soll(p_ret, zr["ok"]) else "nicht über dem Sollwert 95 %"
    st.caption(f"Quelle: {'vorgerechnet' if frozen_row else 'live'}, {zr['ok']} Historien. REML mit t(m − 1) deckt für dieses Merkmal {de_pct(p_ret)}, {urteil}; "
               "in den 20 vorgerechneten Zeilen liegt es zwischen 95,5 und 99,2 %. Die Freiheitsgrade für ein Merkmal innerhalb der Kunden sind nicht m − 1.")

# --- Experiment 4 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Wenn auch das gemischte Modell am Rand liegt")
n_rand = row3["ok"]
formel = MX.p_singulaer_ausgewogen(s.m, s.k, s.rho, "zwischen") if s.su > 0 else MX.p_singulaer_ausgewogen(s.m, s.k, 0.0, "zwischen")
st.markdown(
    f"Liegt σ̂_u² bei 0, heißt die Anpassung **singulär** (ein Rand, kein Absturz). Im **ausgewogenen** Fall mit einem Kundenmerkmal gilt: σ̂_u² (REML) ist genau dann 0, wenn MSB ≤ MSW, "
    f"also mit der Wahrscheinlichkeit P(F(m − 2, m(k − 1)) ≤ 1/(1 + kλ)); für Ihre Einstellung mit gleich großen Kunden {'weniger als 0,1 %' if formel < 0.001 else de_pct(formel)}. Die Vehikel-Kunden sind ungleich groß, die Formel ist hier nur eine Näherung."
)
x1, x2 = st.columns([1, 1])
with x1:
    st.plotly_chart(build_rand(row3, formel), width="stretch", key="exp4_rand")
with x2:
    bed = row3["bedingt"]
    zeilen = []
    for nm, lab in (("rand", "REML am Rand (σ̂_u² = 0)"), ("inner", "REML nicht am Rand")):
        n_, ret, naiv, cr1 = bed[nm]
        zeilen.append({"Historien": lab, "Zahl": n_, "REML t(m−1) in %": 100 * ret / n_ if n_ else None, "naiv in %": 100 * naiv / n_ if n_ else None,
                       "CR1 t(m−1) in %": 100 * cr1 / n_ if n_ else None})
    st.dataframe(pd.DataFrame(zeilen), hide_index=True, width="stretch",
                 column_config={c: st.column_config.NumberColumn(format="%.1f") for c in ("REML t(m−1) in %", "naiv in %", "CR1 t(m−1) in %")})
    st.caption(f"Überdeckung des Intervalls für den Rahmenvertrag, bedingt darauf, ob die REML-Anpassung am Rand lag ({row3['ok']} Historien, {quelle3}).")
kind, text = ev.verdict_rand(row3)
getattr(st, kind)(text)
rang, spalten = _rang(s.m, s.k, s.seed)
st.markdown(
    f"**Wechsel zu festen Kunden-Effekten?** Für ein Merkmal innerhalb der Kunden (Zollhalt) geht das (Überdeckung in Experiment 3). Für den **Rahmenvertrag** geht es nicht: die Spalte ist eine "
    f"Linearkombination der Kunden-Dummies, die Matrix aus {s.m} Dummies und dem Rahmenvertrag hat in dieser Historie Rang {rang} bei {spalten} Spalten, der Koeffizient ist nicht schätzbar."
)

# --- Experiment 5 ----------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 5: Die zwölf Kunden des Generators")
gen = _generator()
e1, e2 = st.columns([1, 1])
with e1:
    st.plotly_chart(build_customers(V.GENERATOR_UK, 0.30), width="stretch", key="exp5_customers")
with e2:
    st.dataframe(pd.DataFrame([
        dict(Größe="Streuung der zwölf Kundeneffekte", realisiert=de(gen["sd"], 3), nominal="0,300"),
        dict(Größe="Intraklassenkorrelation der Import-Container", realisiert=de(gen["icc_real"], 3), nominal=de(gen["icc_nom"], 3)),
        dict(Größe=f"Anteil von {gen['R']} Ziehungen mit Streuung ≥ {de(gen['sd'], 3)}", realisiert=de_pct(gen["anteil"]), nominal="–"),
    ]), hide_index=True, width="stretch")
kind, text = ev.verdict_generator(gen)
getattr(st, kind)(text)
st.toggle("Rahmenvertrag im Modell", key="mim_toggle", help="Fehlt das Kundenmerkmal im Modell, nimmt die Varianzkomponente seine Wirkung auf.")
s = current_settings()
su2 = _su2_live(s.m, s.k, s.su, s.art, ev.live_reps(s.m, s.k, s.reps), s.seed, s.mim)
grid = ev.su2_grid_table()
w1, w2 = st.columns([1, 1])
with w1:
    st.plotly_chart(build_su2(su2["werte"], s.su ** 2), width="stretch", key="exp5_su2")
with w2:
    st.plotly_chart(build_su2_sd(grid), width="stretch", key="exp5_sd")
kind, text = ev.verdict_su2(su2, grid)
getattr(st, kind)(text)
fz = ev.load_frozen().get("su2_merkmal", {})
if fz:
    st.caption(f"Vorgerechnet (m = 12, 30 Container je Kunde, σ_u² = 0,090, {fz['mit']['ok']} Historien): σ̂_u² im Mittel {de(fz['mit']['mean'], 3)} mit Rahmenvertrag im Modell, "
               f"{de(fz['ohne']['mean'], 3)} ohne; erwartet {de(0.09 + 0.0225, 4)} ohne (Varianz der Wirkung des Kundenmerkmals 0,30² · 0,25 = 0,0225 kommt dazu).")

# --- Grenzen ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🚧 Wo die Annahmen enden")
st.markdown(
    """
- **Nur ein Zufallsachsenabschnitt je Kunde.** Zufallssteigungen, mehrere Gruppierungsfaktoren (Kunde und Schiff) und geschachtelte Cluster sind nicht gerechnet.
- **Normale Fehler, konstante Streuung** (der Import-Zweig hat 0,55 auf der log-Skala). Bei Export, Umschlag und Leer ist die Streuung oder der Kundenanteil anders: nicht gemessen.
- **Kunden als Stichprobe:** die Kundeneffekte werden in jeder Historie frisch gezogen. Für einen festen, endlichen Kundenstamm ändert sich die Fragestellung; nicht gemessen.
- **6 bis 48 Kunden.** Unter 6 Kunden ist nicht gemessen; extreme Größenunterschiede (ein Kunde mit der Hälfte der Container) ebenso.
- **Der Rahmenvertrag ist erfunden** (Wirkung −0,30, Verteilung halb/halb, zwei Kunden, stetig). Die Aussagen über Kundenmerkmale gelten für diese drei Verteilungen.
- **Nicht gerechnet:** Satterthwaite- und Kenward-Roger-Freiheitsgrade, CR2, Wild-Cluster-Bootstrap, verallgemeinerte gemischte Modelle (Standgeld, Standtage), gruppierte Kreuzvalidierung.
- **Aus 1000 Historien** (vorgerechnet) bzw. der gewählten Zahl (live) folgt ein Monte-Carlo-Fehler von etwa 0,7 Prozentpunkten bei 95 %; Unterschiede darunter werden nicht gedeutet.
    """
)

# --- Mathematische Formulierung ---------------------------------------------------------------------------------------------------------------

with st.expander("📐 Mathematische Formulierung"):
    st.markdown("**Lesehilfe.** Gegeben sind N Zeilen (Container) in m Kunden und Merkmale x; gesucht ist der Koeffizient β eines Merkmals samt Intervall. Das Modell ist ein Zufallsachsenabschnitt je Kunde.")
    st.markdown(
        """
| Symbol | Bedeutung |
|---|---|
| N, m, k̄ | Zeilen, Kunden, mittlere Zeilen je Kunde (N = m·k̄) |
| y, X, β | log-Verweilzeit, Merkmalsmatrix (mit Achsenabschnitt), Koeffizienten |
| u_k, σ_u² | Kundeneffekt des Kunden k und seine Varianz |
| ε, σ² | Fehler je Zeile und seine Varianz |
| λ, ρ | λ = σ_u²/σ², Intraklassenkorrelation ρ = σ_u²/(σ_u² + σ²) |
| Z | Zuordnung der Zeilen zu Kunden (N × m, Einsen) |
        """
    )
    st.markdown("**Modell** (nur Zufallsachsenabschnitt, normale Fehler, konstante Streuung):")
    st.latex(r"y_{ik}=x_{ik}^{\top}\beta+u_k+\varepsilon_{ik},\qquad u_k\sim N(0,\sigma_u^2),\quad \varepsilon_{ik}\sim N(0,\sigma^2)")
    st.latex(r"V=\sigma^2\left(I+\lambda\,ZZ^{\top}\right),\qquad \lambda=\frac{\sigma_u^2}{\sigma^2},\qquad \rho=\frac{\sigma_u^2}{\sigma_u^2+\sigma^2}")
    st.markdown("**Verallgemeinerte Kleinste Quadrate** bei festem λ und ihre Varianz (die App rechnet in Kundensummen, Woodbury-Formel):")
    st.latex(r"\hat\beta=\left(X^{\top}V^{-1}X\right)^{-1}X^{\top}V^{-1}y,\qquad \operatorname{Var}(\hat\beta)=\left(X^{\top}V^{-1}X\right)^{-1}")
    st.markdown("**Designeffekt** (Varianz der gewöhnlichen Kleinsten Quadrate bei der wahren Kovarianz gegen die naive Formel; im ausgewogenen Fall mit Kundenmerkmal exakt 1 + (k − 1)ρ):")
    st.latex(r"\operatorname{Var}(\hat\beta_{OLS})=\left(X^{\top}X\right)^{-1}X^{\top}VX\left(X^{\top}X\right)^{-1},\qquad \mathrm{Deff}=\frac{\operatorname{Var}(\hat\beta_{OLS})}{(\sigma^2+\sigma_u^2)\,(X^{\top}X)^{-1}}")
    st.latex(r"\mathrm{Deff}=1+(k-1)\rho,\qquad N_{\text{wirksam}}=\frac{N}{\mathrm{Deff}}")
    st.markdown("**Ausgewogener Fall, nur Achsenabschnitt** (k Container je Kunde, MSB und MSW aus der Varianzanalyse); der Schätzer ist 0, wenn die Formel negativ wird (Rand):")
    st.latex(r"\hat\sigma_{u,\mathrm{REML}}^{2}=\max\left(0,\frac{\mathrm{MSB}-\mathrm{MSW}}{k}\right),\qquad \hat\sigma_{u,\mathrm{ML}}^{2}=\max\left(0,\frac{\frac{m-1}{m}\mathrm{MSB}-\mathrm{MSW}}{k}\right)")
    st.latex(r"P\left(\hat\sigma_u^2=0\right)=P\left(F_{m-2,\;m(k-1)}\le\frac{1}{1+k\lambda}\right)\quad\text{(ein Kundenmerkmal im Modell)}")
    st.latex(r"\operatorname{Var}\left(\frac{\mathrm{MSB}-\mathrm{MSW}}{k}\right)=\frac{2}{k^2}\left[\frac{(\sigma^2+k\sigma_u^2)^2}{m-1}+\frac{\sigma^4}{m(k-1)}\right]")
    st.markdown("**Cluster-robuste Varianz CR1** (s_k ist die Summe von x·e über die Zeilen des Kunden k, e das OLS-Residuum):")
    st.latex(r"\widehat V_{CR1}=\frac{m}{m-1}\cdot\frac{N-1}{N-p}\left(X^{\top}X\right)^{-1}\Big(\sum_{k}s_k s_k^{\top}\Big)\left(X^{\top}X\right)^{-1},\qquad s_k=\sum_{i\in k}x_{ik}\,e_{ik}")
    st.markdown("**Ableitung des profilierten Kriteriums bei λ = 0:** ist sie nicht negativ, liegt das Minimum auf dem Rand λ = 0 (so entscheidet die App, nicht über das Gitter allein).")
    st.markdown(
        """
**Konventionen** (damit Lehrbuchzahlen wiederzufinden sind):

| Größe | Hier | Anderswo |
|---|---|---|
| Teiler von σ̂² | REML: N − p; ausgewogen (MSB − MSW)/k | ML teilt durch N: σ̂_u² im MSB-Anteil um den Faktor (m − 1)/m kleiner (Handbeispiel: 20/3 gegen 11) |
| Parametrierung | λ = σ_u²/σ² | lme4 nutzt θ = σ_u/σ = √λ (nach dem Quellenbericht, nicht erneut geöffnet) |
| Intraklassenkorrelation | ρ = σ_u²/(σ_u² + σ²) | je nach Schätzer andere Zahl (Handbeispiel: REML 11/15, ML 5/8) |
| Freiheitsgrade | t(m − 1) für alle Koeffizienten (naiv t(N − p), Kunden-Mittel-Methode t(m − 2)) | für ein Kundenmerkmal im ausgewogenen Fall ist m − 2 exakt; Satterthwaite, Kenward-Roger nicht gerechnet |
| CR1 | Faktor m/(m − 1) · (N − 1)/(N − p) | CR0 ohne Faktor; CR2, CR3 nicht gerechnet |
| „singulär“ | σ̂_u² = 0 (Randlösung λ = 0) | lme4-Sprachgebrauch: die Kovarianzmatrix der Zufallseffekte ist singulär; dasselbe, kein Absturz |
| Zähleinheit | Wiederholung = eine Historie (eine Tabelle aus m Kunden) | „Container“ nur, wo es dasteht |
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
