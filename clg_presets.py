"""Permalink (Begrenzen und Einrasten), Presets und Zufalls-Seed-Knopf (Standardmuster aus dem Demo-Portfolio).

Alle Regler stehen in SETTING_SPECS (auch die beim Experiment im Hauptbereich): so hebt jede Änderung die Markierung des aktiven Presets auf."""

import time
from dataclasses import dataclass

import streamlit as st

import clg_constants as C
from clg_rng import SplitMix64


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: type
    default: object
    lo: float | None = None
    hi: float | None = None
    step: float | None = None
    options: tuple | None = None


SETTING_SPECS = {
    "m_slider": SettingSpec("m", int, C.DEFAULT_M, C.M_MIN, C.M_MAX, 1),
    "k_slider": SettingSpec("k", int, C.DEFAULT_K, C.K_MIN, C.K_MAX, C.K_STEP),
    "su_slider": SettingSpec("su", float, C.DEFAULT_SU, options=C.SU_OPTIONS),
    "art_select": SettingSpec("art", str, C.DEFAULT_ART, options=tuple(C.ARTEN)),
    "seed_input": SettingSpec("seed", int, C.DEFAULT_SEED, 0, C.SEED_MAX),
    "hand_text": SettingSpec("hand", str, C.HAND_BEISPIEL),
    "merk_select": SettingSpec("merk", str, C.DEFAULT_MERKMAL, options=C.MERKMALE),
    "reps_select": SettingSpec("reps", int, C.DEFAULT_REPS, options=C.REPS_OPTIONS),
    "mim_toggle": SettingSpec("mim", int, int(C.DEFAULT_MIM), 0, 1, 1),
}
PRESET_KEYS = {"m": "m_slider", "k": "k_slider", "su": "su_slider", "art": "art_select", "seed": "seed_input", "hand": "hand_text"}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default if state_key != "mim_toggle" else bool(spec.default)


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def snap(spec: SettingSpec, value):
    """Wert auf die Reglergrenzen begrenzen und auf das Raster einrasten (Zahlen) bzw. auf die erlaubten Auswahlwerte prüfen (None, wenn ungültig)."""
    if spec.options is not None:
        if value in spec.options:
            return value
        if all(isinstance(o, (int, float)) for o in spec.options) and isinstance(value, (int, float)):
            return min(spec.options, key=lambda o: abs(o - value))
        return None
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    if spec.step and spec.lo is not None:
        value = spec.lo + round((value - spec.lo) / spec.step) * spec.step
        value = min(value, spec.hi) if spec.hi is not None else value
    return round(value, 10) if isinstance(value, float) else value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, begrenzt und eingerastet."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param not in qp:
            continue
        try:
            value = spec.caster(qp[spec.url_param])
        except (ValueError, TypeError):
            continue
        value = snap(spec, value)
        if value is None:
            continue
        st.session_state[state_key] = bool(value) if state_key == "mim_toggle" else value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(int(value) if isinstance(value, bool) else value)
    except Exception:
        pass


def apply_preset(name):
    """Ein Preset setzt **alle** Regler: zuerst die Standardwerte, dann die Werte des Presets (sonst bliebe ein früher geänderter Regler stehen und das Preset wäre nie aktiv)."""
    for state_key, spec in SETTING_SPECS.items():
        st.session_state[state_key] = bool(spec.default) if state_key == "mim_toggle" else spec.default
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def _gleich(a, b) -> bool:
    return abs(float(a) - float(b)) < 1e-9 if isinstance(a, (int, float)) and isinstance(b, (int, float)) else a == b


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen Standardwert) mit dem Session State."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            _gleich(st.session_state.get(state_key, spec.default), preset.get(preset_key_of.get(state_key), spec.default))
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = SplitMix64(time.time_ns()).below(C.SEED_MAX + 1)


def value_of(key):
    """Wert eines Reglers, auch bevor er in diesem Lauf gezeichnet wurde (Standard, falls noch nie gesetzt)."""
    return st.session_state.get(key, SETTING_SPECS[key].default)
