"""Streamlit GUI for the Yamanote dynamics model.

Two top-level views:

  OPERATIONS  the line as an operator would look at it -- map, train position,
              schedule, rail condition, station busyness. Readable at a glance.
  ANALYSIS    the graphs, the physics, and the validation result, with the
              working shown.

The split exists because the two audiences are different. Operations answers
"where is the train and what is it doing"; Analysis answers "why should I
believe any of this". Mixing them produces a dashboard that serves neither.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Guard before any project or third-party import, so that launching with the
# wrong interpreter produces an instruction rather than a traceback in the
# browser. See scripts/_env.py.
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from _env import ensure  # noqa: E402

ensure(extra=("plotly", "pandas"), streamlit_app="app.py")

import numpy as np  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402

from tokyoline import (  # noqa: E402
    CONDITIONS,
    BrakeProfile,
    DavisCoefficients,
    LoopGeometry,
    TractionModel,
    load_congestion,
    load_ridership,
    load_segments,
    load_spec,
    simulate_circuit,
    simulate_segment,
)
from tokyoline.coasting import pareto_front  # noqa: E402
from tokyoline.fleet import load_fleet, simulatable, to_train_spec  # noqa: E402
from tokyoline.energy import (  # noqa: E402
    AUXILIARY_KW,
    JOULES_PER_KWH,
    REGEN_CUTOFF_KMH,
    REGEN_EFFICIENCY,
    REGEN_RECEPTIVITY,
    circuit_energy,
    segment_energy,
)
from tokyoline.segment import Phase
from tokyoline.validate import (
    decompose,
    dwell_implied_by_margin,
    dwell_implied_by_zero_margin,
    identifiability_locus,
    load_anchors,
    load_practice,
)

st.set_page_config(page_title="Yamanote Line", layout="wide",
                   initial_sidebar_state="expanded")

APP_DIR = Path(__file__).resolve().parent


def apply_theme() -> None:
    """Load the application stylesheet.

    Streamlit's stock styling reads as a template, which undercuts the content.
    Most of assets/theme.css removes its chrome rather than adding decoration.
    """
    css = (APP_DIR / "assets" / "theme.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


apply_theme()

INK = "#1A2027"
MUTED = "#5B6B7A"
GRID = "#EAEEF1"
AXIS = "#D7DEE3"

#: Chart toolbar. Shown always rather than on hover, so "reset axes" is
#: findable, and double-click resets the view on every chart including 3D.
PLOT_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "toggleSpikelines"],
    "doubleClick": "reset",
    "responsive": True,
}

_3D_TYPES = {"scatter3d", "surface", "mesh3d", "cone"}

#: Two roles, two colours, used consistently everywhere a comparison appears.
#: MATH is what this model computes. PRACTICE is what the railway published —
#: the real world. Keeping the pairing identical across every chart is what
#: makes them readable without a legend lookup each time.
MATH = "#1565C0"
PRACTICE = "#2E7D32"


def paired_comparison(items: list[dict], height: int = 260):
    """Small multiples: our math against published practice, in real units.

    One panel per quantity, because the quantities have different units and
    normalising them into a single percentage chart is what made the previous
    version unreadable — a bar at "-18%" tells you nothing about what 18% of
    what. Each panel shows both actual values with the numbers on the bars.
    """
    fig = make_subplots(
        rows=1, cols=len(items), horizontal_spacing=0.075,
        subplot_titles=[i["label"] for i in items],
    )
    for k, it in enumerate(items, start=1):
        fig.add_trace(go.Bar(
            x=["Our math", "Practice"],
            y=[it["model"], it["practice"]],
            marker_color=[MATH, PRACTICE],
            width=0.55,
            text=[f"{it['model']:,.0f}{it['unit']}",
                  f"{it['practice']:,.0f}{it['unit']}"],
            textposition="outside",
            textfont=dict(size=11, color=INK),
            cliponaxis=False,
            hoverinfo="skip", showlegend=False,
        ), row=1, col=k)
        top = max(it["model"], it["practice"]) * 1.32
        fig.update_yaxes(range=[0, top], showticklabels=False, showgrid=False,
                         zeroline=True, row=1, col=k)
        fig.update_xaxes(tickfont=dict(size=10, color=MUTED), row=1, col=k)
    fig.update_layout(height=height, margin=dict(l=4, r=4, t=30, b=4),
                      bargap=0.35)
    for ann in fig.layout.annotations:
        ann.font.size = 11
        ann.font.color = INK
    return fig


def divergence_chart(labels, math_cum, practice_cum, height: int = 380):
    """Cumulative circuit time: physics against the schedule, gap shaded.

    The single most informative comparison in the project. Both curves start
    together at the origin station and separate steadily — the widening band
    between them is dwell, recovery margin and driver caution accumulating stop
    by stop. A pair of totals cannot show that; two curves can.
    """
    # Numeric x, with the station names as tick text. A categorical axis would
    # collapse the two "Shinagawa" entries — origin and loop closure — onto one
    # position, dragging the final point back to x=0 and folding the fill.
    xs = list(range(len(labels)))

    fig = go.Figure()
    # Lower curve first: `tonexty` fills to the PREVIOUS trace, so adding the
    # practice curve second is what shades the band between them rather than
    # flooding down to the axis.
    fig.add_trace(go.Scatter(
        x=xs, y=math_cum, mode="lines", name="Our math (physics minimum)",
        line=dict(color=MATH, width=2.5),
        customdata=labels,
        hovertemplate="%{customdata}<br>math %{y:.1f} min<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=xs, y=practice_cum, mode="lines", name="Practice (published)",
        line=dict(color=PRACTICE, width=2.5),
        fill="tonexty", fillcolor="rgba(46,125,50,0.12)",
        customdata=labels,
        hovertemplate="%{customdata}<br>practice %{y:.1f} min<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin=dict(l=4, r=4, t=10, b=4),
        yaxis_title="Cumulative minutes from the origin station",
        xaxis=dict(tickmode="array", tickvals=xs, ticktext=labels,
                   tickangle=-60, tickfont=dict(size=9),
                   range=[-0.5, len(labels) - 0.5]),
        legend=dict(orientation="h", y=1.1, x=0),
        hovermode="x unified",
    )
    return fig


def chart(fig, height: int | None = None) -> None:
    """Render a figure in the house style, with a persistent toolbar.

    One place applies the light palette, so individual figures do not each
    carry their own colours — that drift is what made the previous build look
    like nineteen unrelated charts.
    """
    fig.update_layout(
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(color=MUTED, size=11, family="Inter, Segoe UI, sans-serif"),
        legend=dict(font=dict(color=MUTED)),
        **({"height": height} if height else {}),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS,
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=AXIS, linecolor=AXIS,
                     tickfont=dict(color=MUTED))
    if any(getattr(t, "type", "") in _3D_TYPES for t in fig.data):
        fig.update_scenes(bgcolor="#FFFFFF")
    for ann in fig.layout.annotations or ():
        if ann.font is not None and ann.font.color == "#8CA0B3":
            ann.font.color = MUTED
    st.plotly_chart(fig, use_container_width=True, config=PLOT_CONFIG)


#: Operating scenarios. Headway and dwell are MODELLED; the peak congestion
#: figure is the one MLIT actually publishes, for the outer loop's critical
#: section. Presenting a scenario rather than five raw sliders is the difference
#: between a tool and a parameter sandbox -- but the tagging has to survive it,
#: so every scenario states which of its numbers is published.
SCENARIOS: dict[str, dict] = {
    "Morning peak": {
        "window": "07:30–09:00",
        "headway_min": 2.5,
        "crowding_pct": 125,
        "dwell_s": 45.0,
        "crowding_source": "published",
        "note": "125% is MLIT's measured figure for Ueno→Okachimachi, 07:43–08:43.",
    },
    "Daytime": {
        "window": "10:00–16:00",
        "headway_min": 4.0,
        "crowding_pct": 60,
        "dwell_s": 32.0,
        "crowding_source": "modelled",
        "note": "Off-peak loading and dwell are modelled — MLIT measures the peak only.",
    },
    "Evening peak": {
        "window": "17:30–19:30",
        "headway_min": 3.0,
        "crowding_pct": 110,
        "dwell_s": 42.0,
        "crowding_source": "modelled",
        "note": "Evening peak is not separately published; modelled below the morning figure.",
    },
    "Late evening": {
        "window": "22:00–00:30",
        "headway_min": 5.0,
        "crowding_pct": 45,
        "dwell_s": 28.0,
        "crowding_source": "modelled",
        "note": "Modelled. Fewer boardings, shorter dwell, wider headway.",
    },
}

PHASE_COLOUR = {
    Phase.ACCEL: "#e4572e",
    Phase.CRUISE: "#17a2b8",
    Phase.COAST: "#f5b700",
    Phase.BRAKE: "#3f51b5",
}

# Degrees of longitude are shorter than degrees of latitude at Tokyo's latitude.
# Without this the loop is drawn noticeably stretched east-west.
LAT_DEG_KM = 111.13
LON_DEG_KM = 90.4
ASPECT = LAT_DEG_KM / LON_DEG_KM


# --------------------------------------------------------------- cached data


@st.cache_resource
def get_fleet():
    return load_fleet()


@st.cache_resource
def get_spec(cls_key: str = "E235-0"):
    """Spec for the selected class.

    The Yamanote's own E235 comes from rolling_stock.json, which carries fuller
    provenance than the comparison fleet file. Other classes are converted from
    fleet.json.
    """
    if cls_key == "E235-0":
        return load_spec()
    return to_train_spec(get_fleet()[cls_key])


@st.cache_resource
def get_segments():
    return load_segments()


@st.cache_resource
def get_geometry():
    return LoopGeometry.build(get_segments())


@st.cache_data
def get_anchors():
    return load_anchors()


@st.cache_data
def get_practice():
    return load_practice()


@st.cache_data
def get_ridership():
    return load_ridership()


@st.cache_data
def get_congestion():
    return load_congestion()


def build(cls_key, load_factor, power_factor, jerk, brake_kmhs):
    spec = get_spec(cls_key)
    davis = DavisCoefficients.estimate_for_emu(
        mass_kg=spec.mass_kg(load_factor),
        n_cars=spec.n_cars,
        frontal_area_m2=spec.frontal_area_m2,
    )
    traction = TractionModel(spec, davis, load_factor, power_factor)
    brake = BrakeProfile(a_max=brake_kmhs / 3.6, jerk=jerk)
    return traction, brake


@st.cache_data(show_spinner=False)
def run_loop(cls_key, load_factor, power_factor, jerk, brake_kmhs, speed_kmh, dt):
    traction, brake = build(cls_key, load_factor, power_factor, jerk, brake_kmhs)
    rows = []
    for s in get_segments():
        r = simulate_segment(traction, brake, s.distance_m,
                             speed_limit_ms=speed_kmh / 3.6, dt=dt)
        rows.append({
            "label": f"{s.from_station} → {s.to_station}",
            "from": s.from_station,
            "distance_m": s.distance_m,
            "run_time_s": r.run_time_s,
            "max_speed_kmh": r.max_speed_kmh,
            "cruise_m": r.phase_distances_m.get("cruise", 0.0),
        })
    return rows


@st.cache_data(show_spinner=False)
def run_segment(cls_key, idx, load_factor, power_factor, jerk, brake_kmhs,
                speed_kmh, dt, coast_start_m):
    traction, brake = build(cls_key, load_factor, power_factor, jerk, brake_kmhs)
    seg = get_segments()[idx]
    r = simulate_segment(traction, brake, seg.distance_m,
                         speed_limit_ms=speed_kmh / 3.6, dt=dt,
                         coast_start_m=coast_start_m)
    return {
        "t": r.t, "x": r.x, "v": r.v, "a": r.a,
        "phase": [p.value for p in r.phase],
        "run_time_s": r.run_time_s, "max_speed_kmh": r.max_speed_kmh,
        "position_error_m": r.position_error_m,
        "phase_times_s": r.phase_times_s,
        "phase_distances_m": r.phase_distances_m,
        "distance_m": seg.distance_m,
        "label": f"{seg.from_station} → {seg.to_station}",
    }


@st.cache_data(show_spinner=False)
def run_circuit(cls_key, load_factor, power_factor, jerk, brake_kmhs, speed_kmh,
                dwell_s, dt):
    traction, brake = build(cls_key, load_factor, power_factor, jerk, brake_kmhs)
    c = simulate_circuit(traction, brake, get_segments(),
                         speed_limit_ms=speed_kmh / 3.6, dwell_s=dwell_s,
                         dt=dt, sample_dt=1.0)
    resistance_kn = np.array([traction.davis(v) for v in c.speed_ms]) / 1000.0
    return {
        "t": c.t, "distance_m": c.distance_m, "speed_kmh": c.speed_kmh,
        "accel_kmh_s": c.accel_kmh_s, "force_n": c.force_n,
        "power_kw": c.power_w / 1000.0, "resistance_kn": resistance_kn,
        "dwelling": c.dwelling, "segment_index": c.segment_index,
        "arrivals": c.arrival_times_s, "departures": c.departure_times_s,
        "run_times": c.run_times_s, "total_time_s": c.total_time_s,
    }


@st.cache_data(show_spinner=False)
def run_energy(cls_key, load_factor, power_factor, jerk, brake_kmhs, speed_kmh, dt,
               dwell_s, receptivity, aux_kw):
    traction, brake = build(cls_key, load_factor, power_factor, jerk, brake_kmhs)
    segs = get_segments()
    results = [
        simulate_segment(traction, brake, s.distance_m,
                         speed_limit_ms=speed_kmh / 3.6, dt=dt)
        for s in segs
    ]
    opts = dict(regen_receptivity=receptivity, auxiliary_kw=aux_kw)
    total, profile = circuit_energy(results, traction, dwell_s=dwell_s, **opts)
    per_segment = [
        segment_energy(r, traction, include_dwell_s=dwell_s, **opts)[0]
        for r in results
    ]
    n_cars = traction.spec.n_cars
    return {
        "traction_kwh": total.traction_j / JOULES_PER_KWH,
        "resistance_kwh": total.resistance_j / JOULES_PER_KWH,
        "brake_kwh": total.brake_system_j / JOULES_PER_KWH,
        "regen_kwh": total.regenerated_j / JOULES_PER_KWH,
        "aux_kwh": total.auxiliary_j / JOULES_PER_KWH,
        "net_kwh": total.net_j / JOULES_PER_KWH,
        "gross_kwh": total.gross_j / JOULES_PER_KWH,
        "regen_saving": total.regen_saving_fraction,
        "kwh_per_km": total.kwh_per_km(),
        "kwh_per_car_km": total.kwh_per_car_km(n_cars),
        "wh_per_pax_km": total.wh_per_passenger_km(
            max(load_factor * traction.spec.capacity, 1.0)),
        "balance_error": total.balance_error_fraction,
        "profile_km": profile.distance_m / 1000.0,
        "profile_traction": profile.traction_j / JOULES_PER_KWH,
        "profile_net": profile.net_j / JOULES_PER_KWH,
        "profile_regen": profile.regenerated_j / JOULES_PER_KWH,
        "seg_labels": [f"{s.from_station} → {s.to_station}" for s in segs],
        "seg_net": [b.net_j / JOULES_PER_KWH for b in per_segment],
        "seg_traction": [b.traction_j / JOULES_PER_KWH for b in per_segment],
        "seg_distance": [s.distance_m for s in segs],
    }


@st.cache_data(show_spinner=False)
def run_pareto(cls_key, idx, load_factor, power_factor, jerk, brake_kmhs, speed_kmh,
               receptivity, aux_kw):
    traction, brake = build(cls_key, load_factor, power_factor, jerk, brake_kmhs)
    seg = get_segments()[idx]
    opts = dict(regen_receptivity=receptivity, auxiliary_kw=aux_kw)
    front = pareto_front(traction, brake, seg.distance_m,
                         speed_limit_ms=speed_kmh / 3.6, dt=0.1, **opts)
    rows = []
    for p in front:
        if p.coast is None or p.slower is None:
            continue
        rows.append({
            "factor": p.time_factor,
            "time_s": p.coast.run_time_s,
            "coast_kwh": p.coast.net_kwh,
            "slower_kwh": p.slower.net_kwh,
            "saving": p.saving_fraction or 0.0,
            "coast_start_m": p.coast.coast_start_m,
            "coast_distance_m": p.coast.coast_distance_m,
            "coast_peak_kmh": p.coast.peak_speed_kmh,
            "slower_peak_kmh": p.slower.peak_speed_kmh,
        })
    return rows


@st.cache_data(show_spinner=False)
def run_decompose(cls_key, load_factor, power_factor, jerk, brake_kmhs, speed_kmh,
                  dwell_s, derate_kmh, round_up_s, published_loop_s):
    traction, brake = build(cls_key, load_factor, power_factor, jerk, brake_kmhs)
    d = decompose(traction, brake, get_segments(), dwell_s=dwell_s,
                  published_loop_s=published_loop_s, line_speed_kmh=speed_kmh,
                  derate_kmh=derate_kmh, round_up_to_s=round_up_s)
    return {
        "rungs": [(r.name, r.cumulative_s, r.delta_s) for r in d.rungs],
        "standard_run_time_s": d.standard_run_time_s,
        "residual_margin_s": d.residual_margin_s,
        "residual_margin_fraction": d.residual_margin_fraction,
    }


def fmt_mmss(seconds: float) -> str:
    s = int(round(seconds))
    return f"{s // 60:d}:{s % 60:02d}"


# ------------------------------------------------------------------ sidebar

segments = get_segments()
geom = get_geometry()
anchors = get_anchors()
practice = get_practice()
ridership = get_ridership()
congestion = get_congestion()
fleet = get_fleet()

PUBLISHED_LOOP_S = anchors["loop_times"]["inner_loop_mean_s"]["value"]
LINE_SPEED_KMH = float(anchors["line_speed_limit_kmh"]["value"])
DWELL = practice["yamanote_dwell_scenarios_seconds"]
CRIT = congestion["critical_sections"]["outer_loop"]

st.sidebar.title("Train")

_runnable = {k: m for k, m in fleet.items() if simulatable(m)[0]}
_blocked = {k: simulatable(m)[1] for k, m in fleet.items() if not simulatable(m)[0]}

#: Livery band colour per class — the stripe that identifies the line it works.
LIVERY = {"E235-0": "#9ACD32", "E233-0": "#F15A22"}


def train_svg(colour: str) -> str:
    """Side elevation of a commuter EMU, drawn rather than photographed.

    A schematic: stainless body, livery band, door and window openings, two
    bogies. It identifies the class at a glance without pulling in a
    copyrighted photograph.
    """
    win = "".join(
        f'<rect x="{x}" y="17" width="20" height="11" rx="2.5" '
        f'fill="#33414D"/>' for x in (44, 96, 148, 200)
    )
    doors = "".join(
        f'<rect x="{x}" y="15" width="13" height="26" rx="1.5" '
        f'fill="#EDF1F3" stroke="#B9C4CC" stroke-width="0.8"/>'
        for x in (74, 126, 178)
    )
    return f"""
<svg viewBox="0 0 250 60" xmlns="http://www.w3.org/2000/svg" role="img">
  <path d="M14 12 h222 a10 10 0 0 1 10 10 v22 a4 4 0 0 1 -4 4 h-234
           a4 4 0 0 1 -4 -4 v-22 a10 10 0 0 1 10 -10 z"
        fill="#F2F5F7" stroke="#93A2AE" stroke-width="1.2"/>
  <path d="M232 13 a10 10 0 0 1 12 9 v6 h-16 z" fill="#33414D"/>
  <rect x="14" y="31" width="232" height="6" fill="{colour}"/>
  {win}{doors}
  <rect x="34" y="46" width="34" height="9" rx="4" fill="#5B6B7A"/>
  <rect x="182" y="46" width="34" height="9" rx="4" fill="#5B6B7A"/>
  <rect x="14" y="44" width="232" height="2" fill="#C7D0D7"/>
</svg>
"""


if "cls_key" not in st.session_state:
    st.session_state.cls_key = list(_runnable)[0]

for key, m in _runnable.items():
    on = st.session_state.cls_key == key
    st.sidebar.markdown(
        f'<div class="train-card{" on" if on else ""}">'
        f'{train_svg(LIVERY.get(key, "#8CA0B3"))}</div>',
        unsafe_allow_html=True,
    )
    if st.sidebar.button(m.display_name, key=f"pick_{key}",
                         use_container_width=True,
                         type="primary" if on else "secondary"):
        st.session_state.cls_key = key
        st.rerun()

cls_key = st.session_state.cls_key
spec = get_spec(cls_key)
member = _runnable[cls_key]

st.sidebar.caption(
    f"{member.formation} · {spec.tare_mass_kg/1000:.1f} t tare · "
    f"{spec.total_power_w/1000:.0f} kW · {spec.max_speed_ms*3.6:.0f} km/h · "
    f"{spec.start_accel_ms2*3.6:.1f} km/h/s"
)
if cls_key != "E235-0":
    st.sidebar.warning(
        f"Running the {member.display_name} over Yamanote geometry — a "
        "counterfactual, not an observation. This class does not work this line."
    )
    if member.decel_is_ambiguous:
        st.sidebar.caption(
            "Its published deceleration carries no service/emergency qualifier, "
            "so braking comparisons against the E235 are not like-for-like."
        )
with st.sidebar.expander(f"{len(_blocked)} classes unavailable"):
    for k, why in _blocked.items():
        st.markdown(f"**{fleet[k].display_name}** — {why}")

st.sidebar.divider()
st.sidebar.title("Service")

scenario_name = st.sidebar.selectbox(
    "Time of day", list(SCENARIOS.keys()), index=1,
    format_func=lambda k: f"{k}  ·  {SCENARIOS[k]['window']}",
)
scen = SCENARIOS[scenario_name]
headway_min = scen["headway_min"]
st.sidebar.caption(scen["note"])

with st.sidebar.expander("Conditions", expanded=True):
    cond_key = st.selectbox(
        "Rail condition", list(CONDITIONS.keys()), index=0,
        format_func=lambda k: CONDITIONS[k].name,
    )
    condition = CONDITIONS[cond_key]
    st.caption(condition.description)

    # Keying on the scenario means changing time of day resets these to that
    # scenario's values, while still allowing a manual override afterwards.
    crowding_pct = st.slider(
        "Crowding (congestion rate %)", 0, 200, int(scen["crowding_pct"]), 5,
        key=f"crowd_{scenario_name}",
        help=f"MLIT measures {CRIT['congestion_rate_percent']}% on the outer "
             f"loop's critical section ({CRIT['from']}→{CRIT['to']}, peak hour). "
             "100% = every passenger can sit or hold a strap.",
    )
    load_factor = crowding_pct / 100.0

with st.sidebar.expander("Driving", expanded=True):
    service_brake_kmhs = st.slider(
        "Service brake rate (km/h/s)", 2.0, spec.service_decel_ms2 * 3.6,
        spec.service_decel_ms2 * 3.6, 0.1,
        help=f"{spec.service_decel_ms2*3.6:.1f} is this class's published maximum "
             "service rate. Drivers do not use maximum service braking into every "
             "platform, so lower values are more realistic.",
    )
    service_brake_ms2 = service_brake_kmhs / 3.6
    brake_ms2 = condition.effective_brake_rate_ms2(service_brake_ms2, 0.0)
    brake_kmhs = brake_ms2 * 3.6

    if condition.is_adhesion_limited(service_brake_ms2, 0.0):
        st.error(
            f"Adhesion-limited: grip caps braking at {brake_kmhs:.2f} km/h/s, "
            f"below the {service_brake_kmhs:.1f} km/h/s service rate."
        )

    jerk = st.slider("Comfort jerk limit (m/s³)", 0.3, 1.2, 0.75, 0.05,
                     help="Modelled. Literature puts the comfortable range for "
                          "standing passengers at 0.5–1.0 m/s³.")
    power_factor = st.slider(
        "Power factor", 0.7, 1.5, 1.0, 0.05,
        help="Multiplier on the published one-hour rating. Leave at 1.0 for the "
             "headline result — raising it is a sensitivity test, not a fit.")
    speed_kmh = st.slider("Speed limit (km/h)", 60.0, 90.0, LINE_SPEED_KMH, 1.0)
    dwell_s = st.slider(
        "Mean dwell (s)", 15.0, 55.0, float(scen["dwell_s"]), 1.0,
        key=f"dwell_{scenario_name}",
        help="NOT published by JR East. Inferred at 39–50 s — see Validation.")

with st.sidebar.expander("Energy", expanded=False):
    receptivity = st.slider(
        "Regen receptivity", 0.0, 1.0, REGEN_RECEPTIVITY, 0.05,
        help="Fraction of returned energy another train actually absorbs. NOT "
             "published — the least defensible number in the model, and the net "
             "result moves by nearly 3x across this range.",
    )
    aux_kw = st.slider(
        "Auxiliary load (kW)", 0.0, 250.0, AUXILIARY_KW, 10.0,
        help="HVAC, lighting, compressors. Modelled. Runs during dwell too.",
    )
    st.caption(
        f"Efficiency fixed at {REGEN_EFFICIENCY:.2f}, cutoff at "
        f"{REGEN_CUTOFF_KMH:.0f} km/h. Both modelled. JR East publishes a 回生率 "
        "of 59.0% for this class on this line; the model gives 54.7% here, and "
        "is deliberately left untuned because that ratio's denominator could "
        "not be verified."
    )

with st.sidebar.expander("Numerics", expanded=False):
    dt = st.select_slider("Integration step (s)", [0.2, 0.1, 0.05, 0.02], value=0.1)
    st.caption("Circuit time varies by under 10 ms across this range. The energy "
               "balance residual is first-order in dt.")

traction, brake = build(cls_key, load_factor, power_factor, jerk, brake_kmhs)

st.sidebar.caption(
    f"Effective mass {spec.effective_mass_kg(load_factor)/1000:.0f} t · "
    f"max effort {traction.max_effort_n/1000:.0f} kN · "
    f"stop from 90 {brake.distance_to_stop(25.0):.0f} m"
)


# --------------------------------------------------------------------- views

st.markdown(
    f"""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:2px;">
      <div style="width:6px;height:34px;background:#9ACD32;border-radius:3px;"></div>
      <div>
        <div style="font-size:1.45rem;font-weight:660;letter-spacing:-0.01em;">
          山手線 &nbsp;Yamanote Line</div>
        <div style="font-size:0.78rem;color:#5B6B7A;letter-spacing:0.02em;">
          {scenario_name} · {scen['window']} · {len(segments)} stations ·
          34.5 km · {spec.name.split(',')[0]}</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Results and limits", expanded=False):
    fa, fb = st.columns(2)
    with fa:
        st.markdown(
            "**Results**\n\n"
            "- Minimum circuit 2,309 s; published 3,948 s.\n"
            "- Dwell inferred at 39–50 s per station, by constraining recovery "
            "margin to documented MLIT practice.\n"
            "- 92% of traction work is dissipated in braking.\n"
            "- 10% more run time reduces circuit energy by about a third. "
            "Coasting adds 2–4%."
        )
    with fb:
        st.markdown(
            "**Limits**\n\n"
            "- Dwell is inferred, not measured. Circuit time = run + margin + "
            "dwell is one equation in two unknowns.\n"
            "- Signal and ATC restrictions are not public.\n"
            "- Davis coefficients are modelled; no published set exists for "
            "this class.\n"
            "- Regeneration receptivity is unpublished and moves net energy by "
            "a factor of three."
        )
    st.caption(
        "Parameters are tagged published, derived, modelled, bounded or "
        "unknown in the Provenance tab."
    )

view_overview, view_analysis = st.tabs(["Line", "Technical"])

# Departures, the comparison and the map/3D all render into the same tab.
# Re-entering one container with several `with` blocks appends in order, so the
# three sections merge into a single page without re-indenting their bodies.
view_board = view_truth = view_net = view_overview


# =================================================================== BOARD

def journey_time_s(from_idx: int, to_idx: int, run_times: list[float],
                   dwell: float, outer: bool) -> float:
    """Modelled time from one station to another, including intermediate dwells.

    Segment i runs from station i to station i+1 in the outer-loop direction, so
    the inner loop traverses the same segments backwards. Dwell is counted at
    every intermediate stop but not at the destination.
    """
    n = len(run_times)
    total = 0.0
    i = from_idx
    hops = 0
    while i != to_idx and hops <= n:
        if outer:
            total += run_times[i % n]
            i = (i + 1) % n
        else:
            total += run_times[(i - 1) % n]
            i = (i - 1) % n
        hops += 1
        if i != to_idx:
            total += dwell
    return total


with view_board:
    rows_b = run_loop(cls_key, load_factor, power_factor, jerk, brake_kmhs,
                      speed_kmh, dt)
    run_times = [r["run_time_s"] for r in rows_b]

    now = datetime.now(timezone(timedelta(hours=9)))
    circuit_s = sum(run_times) + dwell_s * len(run_times)

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Tokyo time", now.strftime("%H:%M"), scenario_name, delta_color="off")
    b2.metric("Headway", f"{headway_min:.1f} min", "modelled", delta_color="off")
    b3.metric("Circuit, modelled", fmt_mmss(circuit_s),
              f"published {fmt_mmss(PUBLISHED_LOOP_S)}", delta_color="off")
    b4.metric("Crowding", f"{crowding_pct}%",
              scen["crowding_source"], delta_color="off")

    st.warning(
        "Modelled schedule, not a live feed. Departure times come from the "
        "physics model plus an assumed headway and dwell. JR East real-time "
        "data requires an ODPT consumer key, which this project does not have."
    )

    sel_col, dest_col = st.columns([1, 1])
    with sel_col:
        board_idx = st.selectbox(
            "Station", range(len(segments)),
            format_func=lambda i: segments[i].from_station,
            index=segments.index(next(s for s in segments
                                      if s.from_station == "Shinjuku")),
        )
    with dest_col:
        dest_idx = st.selectbox(
            "Journey time to", range(len(segments)),
            format_func=lambda i: segments[i].from_station,
            index=segments.index(next(s for s in segments
                                      if s.from_station == "Tokyo")),
        )

    station_name = segments[board_idx].from_station

    def load_badge(pct: int) -> tuple[str, str]:
        if pct < 80:
            return "load-easy", "Seats"
        if pct < 120:
            return "load-mid", "Standing"
        return "load-heavy", "Crowded"

    cls_load, label_load = load_badge(crowding_pct)

    # Departures are anchored to the clock at the modelled headway. The phase is
    # arbitrary -- there is no published departure phase to align to -- so the
    # minutes are illustrative of frequency, not of any specific service.
    head_s = headway_min * 60.0
    base = (now.minute * 60 + now.second)
    first_offset = head_s - (base % head_s)

    outer_via = "via Shibuya · Shinagawa"
    inner_via = "via Ikebukuro · Ueno"

    # The model runs the circuit faster than the railway schedules it. Scaling
    # journey times by published/modelled turns a physics minimum into an
    # estimate calibrated against the one timetable figure published to the
    # second. Both are shown — the difference IS the operational overhead.
    calib = PUBLISHED_LOOP_S / circuit_s if circuit_s > 0 else 1.0

    dest_name = segments[dest_idx].from_station
    rows_html = [
        '<div class="board">',
        '<div class="board-row head"><div>Departs</div><div>In</div>'
        '<div>Direction</div><div>Loading</div>'
        f'<div>To {dest_name}</div></div>',
    ]
    for k in range(8):
        outer = (k % 2 == 0)
        secs = first_offset + k * head_s / 2
        t = now + timedelta(seconds=secs)
        jt = journey_time_s(board_idx, dest_idx, run_times, dwell_s, outer)
        mins = int(secs // 60)
        in_txt = "due" if mins < 1 else f"{mins} min"
        soon = " soon" if mins < 2 else ""
        direction = "Outer loop 外回り" if outer else "Inner loop 内回り"
        via = outer_via if outer else inner_via
        rows_html.append(
            f'<div class="board-row">'
            f'<div class="board-time">{t.strftime("%H:%M")}</div>'
            f'<div class="board-in{soon}">{in_txt}</div>'
            f'<div><div class="board-dest">{direction}</div>'
            f'<div class="board-sub">{via}</div></div>'
            f'<div class="board-load {cls_load}">{label_load}</div>'
            f'<div><div class="jt-likely">{fmt_mmss(jt * calib)} likely</div>'
            f'<div class="jt-phys">{fmt_mmss(jt)} physics minimum</div></div>'
            f'</div>'
        )
    rows_html.append("</div>")
    st.markdown("".join(rows_html), unsafe_allow_html=True)

    cal1, cal2 = st.columns([2, 3])
    with cal1:
        st.metric("Schedule calibration", f"×{calib:.2f}",
                  "published ÷ modelled circuit", delta_color="off")
    with cal2:
        st.markdown(
            f"*Physics minimum* is the model's run time. *Likely* multiplies it "
            f"by {calib:.2f} — published circuit time "
            f"({fmt_mmss(PUBLISHED_LOOP_S)}) over modelled "
            f"({fmt_mmss(circuit_s)}). That is the only timetable figure "
            "published to the second. The difference between the columns is "
            "dwell, recovery margin, driver caution and signalling."
        )

    st.divider()
    st.markdown("##### Modelled journey time from " + station_name)
    jt_outer = [journey_time_s(board_idx, i, run_times, dwell_s, True)
                for i in range(len(segments))]
    jt_inner = [journey_time_s(board_idx, i, run_times, dwell_s, False)
                for i in range(len(segments))]
    best = [min(a, b) for a, b in zip(jt_outer, jt_inner)]
    order = sorted(range(len(segments)), key=lambda i: best[i])

    fig_j = go.Figure(go.Bar(
        x=[segments[i].from_station for i in order],
        y=[best[i] / 60.0 for i in order],
        marker_color=["#9ACD32" if jt_outer[i] <= jt_inner[i] else "#4FA3D1"
                      for i in order],
        hovertemplate="%{x}<br>%{y:.1f} min<extra></extra>",
    ))
    fig_j.update_layout(
        height=340, yaxis_title="Minutes (faster direction)",
        xaxis_tickangle=-60, margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8CA0B3"),
    )
    chart(fig_j)
    st.caption("Green: faster on the outer loop. Blue: faster on the inner loop.")


# ============================================================ MODEL VS REALITY

with view_truth:
    st.markdown("##### Physics against the timetable, station by station")

    _rows = run_loop(cls_key, load_factor, power_factor, jerk, brake_kmhs,
                     speed_kmh, dt)
    _rt = [r["run_time_s"] for r in _rows]
    _names = [s.from_station for s in segments] + [segments[0].from_station]

    _math_cum, _acc = [0.0], 0.0
    for i, rt in enumerate(_rt):
        _acc += rt + (dwell_s if i < len(_rt) - 1 else 0.0)
        _math_cum.append(_acc / 60.0)

    # Practice is the published circuit time distributed across segments in
    # proportion to the model's own run times. The total is real; the split is
    # an assumption, stated below the chart rather than buried.
    _scale = PUBLISHED_LOOP_S / _acc if _acc else 1.0
    _prac_cum = [v * _scale for v in _math_cum]

    chart(divergence_chart(_names, _math_cum, _prac_cum))
    st.markdown(
        f"The curves separate by {_prac_cum[-1] - _math_cum[-1]:.0f} minutes "
        "over one circuit. The band is dwell, recovery margin, driver caution "
        "and signalling, accumulated stop by stop."
    )
    st.caption(
        f"The circuit total is published ({fmt_mmss(PUBLISHED_LOOP_S)}); its "
        "split across the 30 segments is not, because public timetables are "
        "minute-rounded. The practice curve distributes the published total in "
        "proportion to modelled run times: endpoints measured, path assumed."
    )

    st.divider()
    st.markdown("##### Head to head")

    e_t = run_energy(cls_key, load_factor, power_factor, jerk, brake_kmhs,
                     speed_kmh, dt, dwell_s, receptivity, aux_kw)
    loop_rows = run_loop(cls_key, load_factor, power_factor, jerk, brake_kmhs,
                         speed_kmh, dt)
    model_min = sum(r["run_time_s"] for r in loop_rows)

    # Only quantities with a genuine PUBLISHED point value belong on this chart.
    # Mean dwell is deliberately absent: the model's dwell is a slider set from
    # our own inference, so charting it against that inference would be circular
    # and would dress an assumption up as a validation.
    checks = [
        {
            "quantity": "Regeneration rate",
            "model": e_t["regen_kwh"] / e_t["traction_kwh"] * 100,
            "published": 59.0,
            "unit": "%",
            "source": "JR East 回生率, E235 on this line",
        },
        {
            "quantity": "Circuit time",
            "model": model_min + dwell_s * len(loop_rows),
            "published": float(PUBLISHED_LOOP_S),
            "unit": " s",
            "source": "published inner-loop mean",
        },
    ]
    # Traction power and starting acceleration are deliberately NOT here either.
    # They are model INPUTS read straight from the published specification, so
    # comparing them against their own source returns 0% by construction. Two
    # green bars that can never be anything else would imply validation the
    # model has not earned.

    chart(paired_comparison([
        {"label": "Circuit time", "model": checks[1]["model"],
         "practice": checks[1]["published"], "unit": " s"},
        {"label": "Regeneration", "model": checks[0]["model"],
         "practice": checks[0]["published"], "unit": "%"},
    ]))
    st.caption(
        "Only model outputs with a published counterpart appear here. Dwell is "
        "excluded because the model's dwell is itself an inference. Traction "
        "power and starting acceleration are excluded because they are inputs "
        "read from the specification: comparing them to their own source "
        "returns zero difference by construction."
    )

    band_lo, band_hi = 2.0, 4.0
    per_car = e_t["kwh_per_car_km"]
    inside = band_lo <= per_car <= band_hi
    st.markdown("##### Energy against a published *range*")
    st.markdown(
        f"Specific consumption is **{per_car:.2f} kWh per car-km** against a "
        f"published heavy-metro band of **{band_lo:.0f}–{band_hi:.0f}**. "
        + ("That sits inside the band." if inside else
           f"**That sits {'below' if per_car < band_lo else 'above'} the band**, "
           "not inside it.")
    )
    if not inside and per_car < band_lo:
        st.warning(
            f"{100*(1 - per_car/band_lo):.0f}% under the band at the assumed "
            f"receptivity of {receptivity:.2f}. Receptivity of 0.3–0.5 moves it "
            "inside; the 回生率 cross-check implies about 0.76. The two "
            "published anchors disagree on the same parameter and this project "
            "does not resolve the conflict."
        )

    st.divider()
    g1, g2 = st.columns(2)
    with g1:
        st.markdown("##### Agreement")
        st.markdown(
            f"- Regeneration: {e_t['regen_kwh']/e_t['traction_kwh']*100:.1f}% "
            "against a published 59.0%.\n"
            "- Adhesion never binds, as the 6/11 powered-axle ratio implies.\n"
            "- Braking distances follow from published rates, unfitted."
        )
    with g2:
        st.markdown("##### Disagreement")
        st.markdown(
            f"- Circuit time: {model_min:.0f} s modelled against "
            f"{PUBLISHED_LOOP_S} s scheduled.\n"
            "- Specific energy: 1.82 kWh/car-km against a 2–4 band.\n"
            "- Coasting: 2–4% saving against 12–15.7% reported by JR East."
        )

    st.markdown(
        "**Summary.** The model reproduces train behaviour and not railway "
        "operation. The circuit-time gap is dwell, recovery margin, driver "
        "caution and signalling, none of which the physics contains."
    )


# ======================================================================= NETWORK

with view_net:
    circuit = run_circuit(cls_key, load_factor, power_factor, jerk, brake_kmhs,
                          speed_kmh, dwell_s, dt)
    rows = run_loop(cls_key, load_factor, power_factor, jerk, brake_kmhs,
                    speed_kmh, dt)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Circuit time", fmt_mmss(circuit["total_time_s"]))
    c2.metric("Published", fmt_mmss(PUBLISHED_LOOP_S),
              f"{circuit['total_time_s'] - PUBLISHED_LOOP_S:+.0f} s",
              delta_color="off")
    c3.metric("Rail", condition.name)
    c4.metric("Crowding", f"{crowding_pct}%",
              f"{(load_factor - 1.0) * spec.capacity * spec.passenger_mass_kg / 1000:+.0f} t",
              delta_color="off")
    c5.metric("Top speed", f"{max(r['max_speed_kmh'] for r in rows):.0f} km/h")

    # Station display data, shared by the live view and the 2D map below.
    board = {r.name_en: r.boardings_2024 for r in ridership.itertuples()}
    b = np.array([board[n] for n in geom.names], dtype=float)
    sizes = 9 + 30 * (np.sqrt(b) - np.sqrt(b.min())) / (
        np.sqrt(b.max()) - np.sqrt(b.min()))
    hover = [f"<b>{n}</b><br>{board[n]:,} boardings/day" for n in geom.names]

    # ---- live 3D track view with synchronised instrument panels ----------
    st.markdown("##### Live run")

    N_FRAMES = 200
    ct = circuit["t"]
    c_speed = circuit["speed_kmh"]
    c_accel = circuit["accel_kmh_s"]
    c_force = circuit["force_n"]
    c_power = circuit["power_kw"]
    c_res = circuit["resistance_kn"]
    c_dist = circuit["distance_m"]

    f_t = np.linspace(0, circuit["total_time_s"], N_FRAMES, endpoint=False)
    f_d = np.interp(f_t, ct, c_dist)
    f_v = np.interp(f_t, ct, c_speed)
    f_a = np.interp(f_t, ct, c_accel)
    f_p = np.interp(f_t, ct, c_power)
    f_r = np.interp(f_t, ct, c_res)
    f_f = np.interp(f_t, ct, c_force) / 1000.0

    # Track polyline, densely sampled so the speed colouring is smooth.
    track_d = np.linspace(0, geom.total_m, 900, endpoint=False)
    track_pos = [geom.position_at(d) for d in track_d]
    track_lat = np.array([p[0] for p in track_pos])
    track_lon = np.array([p[1] for p in track_pos])
    track_v = np.interp(track_d, c_dist, c_speed)

    # The train is 214 m against a 34.5 km loop — about two pixels at this
    # zoom. Drawn 4x long so it reads as a train; the caption says so.
    CAR_GAP_M = 200.0
    N_CARS = 5

    def train_xyz(d0: float) -> tuple[list, list, list]:
        xs, ys, zs_ = [], [], []
        for k in range(N_CARS):
            p = geom.position_at(d0 - k * CAR_GAP_M)
            xs.append(p[1])
            ys.append(p[0])
            zs_.append(0.0)
        return xs, ys, zs_

    fig3 = make_subplots(
        rows=5, cols=2, column_widths=[0.60, 0.40],
        horizontal_spacing=0.07, vertical_spacing=0.055,
        specs=[[{"type": "scene", "rowspan": 5}, {"type": "xy"}],
               [None, {"type": "xy"}],
               [None, {"type": "xy"}],
               [None, {"type": "xy"}],
               [None, {"type": "xy"}]],
        subplot_titles=(
            "",
            f"SPEED  km/h　·　dotted: published line limit {speed_kmh:.0f}",
            f"ACCELERATION  km/h/s　·　dotted: published "
            f"{spec.start_accel_ms2*3.6:.1f} start / −{service_brake_kmhs:.1f} brake",
            "TRACTIVE EFFORT  kN　·　no published curve for this class",
            f"POWER AT THE WHEEL  kW　·　dotted: published "
            f"{traction.power_w/1000:.0f} kW one-hour rating",
            "RUNNING RESISTANCE  kN　·　modelled — no published value",
        ),
    )

    # --- 3D scene -----------------------------------------------------
    # Single-hue ramp: pale where the train is slow, deep green where it is
    # fast. A rainbow scale reads as decoration; one hue reads as a quantity.
    SPEED_SCALE = [[0.0, "#DDE9CB"], [0.5, "#9ACD32"], [1.0, "#33691E"]]
    fig3.add_trace(go.Scatter3d(
        x=track_lon, y=track_lat, z=np.zeros_like(track_lat), mode="lines",
        line=dict(color=track_v, colorscale=SPEED_SCALE, width=11,
                  cmin=0, cmax=max(speed_kmh, 1)),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    pil_x, pil_y, pil_z = [], [], []
    for lo, la in zip(geom.lon, geom.lat):
        pil_x += [lo, lo, None]
        pil_y += [la, la, None]
        pil_z += [0.0, 12.0, None]
    fig3.add_trace(go.Scatter3d(
        x=pil_x, y=pil_y, z=pil_z, mode="lines",
        line=dict(color="rgba(90,107,122,0.30)", width=1.5),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)

    fig3.add_trace(go.Scatter3d(
        x=geom.lon, y=geom.lat, z=np.full(len(geom.lat), 12.0),
        mode="markers+text",
        marker=dict(size=4, color="#FFFFFF",
                    line=dict(color=INK, width=1.5)),
        text=geom.names, textposition="top center",
        textfont=dict(size=8, color=MUTED),
        hovertext=hover, hoverinfo="text", showlegend=False,
    ), row=1, col=1)

    tx, ty, tz = train_xyz(f_d[0])
    fig3.add_trace(go.Scatter3d(
        x=tx, y=ty, z=tz, mode="lines+markers",
        line=dict(color="#1565C0", width=10),
        marker=dict(size=6, color=["#0D47A1"] + ["#1565C0"] * (N_CARS - 1),
                    symbol="square"),
        hoverinfo="skip", showlegend=False,
    ), row=1, col=1)
    TRAIN_TRACE = 3

    # --- instrument panels --------------------------------------------
    panels = [
        (1, c_speed, f_v, "#9ACD32", "km/h",
         [(speed_kmh, f"line limit {speed_kmh:.0f}", "#E5484D")]),
        (2, c_accel, f_a, "#4FA3D1", "km/h/s",
         [(spec.start_accel_ms2 * 3.6, "published start accel", "#3FB950"),
          (-service_brake_kmhs, "published max service brake", "#E5484D")]),
        (3, np.interp(ct, ct, c_force) / 1000.0, f_f, "#E4A11B", "kN", []),
        (4, c_power, f_p, "#B98AE0", "kW",
         [(traction.power_w / 1000.0, "one-hour rating", "#E5484D")]),
        (5, c_res, f_r, "#7FB6A6", "kN", []),
    ]

    marker_traces: list[int] = []
    for row, series, fseries, colour, unit, refs in panels:
        fig3.add_trace(go.Scatter(
            x=ct, y=series, mode="lines",
            line=dict(color=colour, width=1.6),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=2)
        # Reference lines are drawn as traces rather than shapes: add_hline
        # with row/col inspects every trace for an `xaxis` property, which
        # Scatter3d does not have, and raises on a mixed 3D/2D figure.
        for value, label, rcol in refs:
            fig3.add_trace(go.Scatter(
                x=[float(ct[0]), float(ct[-1])], y=[value, value], mode="lines",
                line=dict(color=rcol, width=1, dash="dot"),
                hoverinfo="skip", showlegend=False,
            ), row=row, col=2)
        fig3.add_trace(go.Scatter(
            x=[f_t[0]], y=[fseries[0]], mode="markers",
            marker=dict(size=9, color="#FFFFFF",
                        line=dict(color=colour, width=2.5)),
            hoverinfo="skip", showlegend=False,
        ), row=row, col=2)
        marker_traces.append(len(fig3.data) - 1)
        fig3.update_yaxes(title_text=unit, row=row, col=2,
                          title_font=dict(size=9), tickfont=dict(size=8),
                          gridcolor="rgba(140,160,179,0.15)", zeroline=False)
        fig3.update_xaxes(row=row, col=2, showticklabels=(row == 5),
                          tickfont=dict(size=8),
                          gridcolor="rgba(140,160,179,0.10)", zeroline=False)

    fig3.update_xaxes(title_text="seconds into the circuit", row=5, col=2,
                      title_font=dict(size=9))

    fig3.frames = [
        go.Frame(
            name=str(i),
            data=(
                [go.Scatter3d(x=train_xyz(f_d[i])[0], y=train_xyz(f_d[i])[1],
                              z=train_xyz(f_d[i])[2])]
                + [go.Scatter(x=[f_t[i]], y=[s[i]])
                   for s in (f_v, f_a, f_f, f_p, f_r)]
            ),
            traces=[TRAIN_TRACE] + marker_traces,
        )
        for i in range(N_FRAMES)
    ]

    _ax3 = dict(title="", showticklabels=False, showbackground=False,
                showgrid=False, zeroline=False, visible=False)
    fig3.update_layout(
        height=760, showlegend=False,
        margin=dict(l=0, r=6, t=26, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8CA0B3", size=10),
        scene=dict(
            xaxis=_ax3, yaxis=_ax3,
            zaxis=dict(**{**_ax3, "range": [0, 260]}),
            aspectmode="manual", aspectratio=dict(x=1, y=ASPECT, z=0.28),
            camera=dict(eye=dict(x=1.25, y=-1.25, z=1.05),
                        up=dict(x=0, y=0, z=1)),
            bgcolor="rgba(0,0,0,0)",
        ),
        updatemenus=[dict(
            type="buttons", showactive=False, x=0.0, y=0.0, xanchor="left",
            bgcolor="#FFFFFF", bordercolor="#D7DEE3",
            font=dict(color=INK, size=11),
            buttons=[
                dict(label="Run", method="animate",
                     args=[None, dict(frame=dict(duration=70, redraw=True),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate")]),
            ],
        )],
    )
    for ann in fig3.layout.annotations:
        ann.font.size = 9.5
        ann.font.color = "#8CA0B3"
        ann.xanchor = "left"
        ann.x = ann.x - 0.185 if ann.x > 0.5 else ann.x

    chart(fig3)
    st.caption(
        "Track colour is speed at each point on the loop. Dotted lines are "
        "published figures: line speed limit, starting acceleration, maximum "
        "service brake rate, one-hour power rating. Tractive effort and running "
        "resistance have no published curve for this class and show the model "
        f"alone. The train is drawn "
        f"{N_CARS * CAR_GAP_M / spec.train_length_m:.0f}× its real length for "
        "visibility."
    )

    st.divider()
    left, right = st.columns([3, 2])

    # ---- map -------------------------------------------------------------
    with left:
        dim = st.radio("View", ["2D map", "3D speed profile"], horizontal=True,
                       label_visibility="collapsed")

        board = {r.name_en: r.boardings_2024 for r in ridership.itertuples()}
        b = np.array([board[n] for n in geom.names], dtype=float)
        sizes = 9 + 30 * (np.sqrt(b) - np.sqrt(b.min())) / (np.sqrt(b.max()) - np.sqrt(b.min()))

        path_lat, path_lon = geom.closed_path()
        n_frames = 160
        frame_t = np.linspace(0, circuit["total_time_s"], n_frames, endpoint=False)
        frame_d = np.interp(frame_t, circuit["t"], circuit["distance_m"])
        frame_v = np.interp(frame_t, circuit["t"], circuit["speed_kmh"])
        pos = [geom.position_at(d) for d in frame_d]
        train_lat = np.array([p[0] for p in pos])
        train_lon = np.array([p[1] for p in pos])

        hover = [f"<b>{n}</b><br>{board[n]:,} boardings/day" for n in geom.names]

        if dim == "2D map":
            fig = go.Figure()
            # Transit-diagram styling: one heavy line in the operator's own
            # colour, with a darker casing beneath so it reads as a route rather
            # than a plotted series.
            fig.add_trace(go.Scatter(
                x=path_lon, y=path_lat, mode="lines",
                line=dict(color="#5A8020", width=13), hoverinfo="skip",
                showlegend=False,
            ))
            fig.add_trace(go.Scatter(
                x=path_lon, y=path_lat, mode="lines",
                line=dict(color="#9ACD32", width=9), name="Line", hoverinfo="skip",
            ))
            # Highlight the section where congestion is actually measured.
            try:
                ci = geom.names.index(CRIT["from"])
                cj = geom.names.index(CRIT["to"])
                fig.add_trace(go.Scatter(
                    x=[geom.lon[ci], geom.lon[cj]], y=[geom.lat[ci], geom.lat[cj]],
                    mode="lines", line=dict(color="#C62828", width=9),
                    name=f"Peak critical section ({CRIT['congestion_rate_percent']}%)",
                    hovertemplate=f"{CRIT['from']}→{CRIT['to']}<br>"
                                  f"{CRIT['congestion_rate_percent']}% at "
                                  f"{CRIT['measurement_window']}<extra></extra>",
                ))
            except ValueError:
                pass
            fig.add_trace(go.Scatter(
                x=geom.lon, y=geom.lat, mode="markers+text", name="Stations",
                marker=dict(size=sizes, color="#FFFFFF",
                            line=dict(color=INK, width=2)),
                text=geom.names, textposition="top center",
                textfont=dict(size=9.5, color=INK),
                hovertext=hover, hoverinfo="text",
            ))
            fig.add_trace(go.Scatter(
                x=[train_lon[0]], y=[train_lat[0]], mode="markers", name="Train",
                marker=dict(size=17, color="#1565C0", symbol="circle",
                            line=dict(color="#FFFFFF", width=3)),
            ))
            frames = [
                go.Frame(
                    data=[go.Scatter(x=[train_lon[i]], y=[train_lat[i]])],
                    traces=[len(fig.data) - 1], name=str(i),
                )
                for i in range(n_frames)
            ]
            fig.frames = frames
            fig.update_yaxes(scaleanchor="x", scaleratio=ASPECT)
            fig.update_layout(
                height=620, showlegend=False,
                xaxis=dict(visible=False), yaxis=dict(visible=False),
                margin=dict(l=0, r=0, t=20, b=0),
            )
        else:
            zs = np.interp(circuit["distance_m"], circuit["distance_m"],
                           circuit["speed_kmh"])
            trace_pos = [geom.position_at(d) for d in circuit["distance_m"]]
            tl = np.array([p[0] for p in trace_pos])
            tn = np.array([p[1] for p in trace_pos])

            fig = go.Figure()

            # Ground trace of the loop, so the speed ribbon reads as height
            # above a recognisable plan rather than floating in space.
            fig.add_trace(go.Scatter3d(
                x=path_lon, y=path_lat, z=np.zeros_like(path_lat), mode="lines",
                line=dict(color="#c3ced9", width=3),
                name="Ground track", hoverinfo="skip",
            ))

            # Vertical drop lines at each station, drawn as one trace with NaN
            # breaks rather than 30 separate traces.
            drop_x, drop_y, drop_z = [], [], []
            for lon_i, lat_i in zip(geom.lon, geom.lat):
                drop_x += [lon_i, lon_i, None]
                drop_y += [lat_i, lat_i, None]
                drop_z += [0.0, speed_kmh, None]
            fig.add_trace(go.Scatter3d(
                x=drop_x, y=drop_y, z=drop_z, mode="lines",
                line=dict(color="rgba(160,175,190,0.35)", width=1),
                name="Stations", hoverinfo="skip",
            ))

            fig.add_trace(go.Scatter3d(
                x=tn, y=tl, z=zs, mode="lines",
                line=dict(color=zs, colorscale="Turbo", width=8,
                          colorbar=dict(title="km/h", x=1.02, len=0.7)),
                name="Speed", hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter3d(
                x=geom.lon, y=geom.lat, z=np.zeros_like(geom.lat), mode="markers+text",
                marker=dict(size=sizes / 4 + 3, color=b, colorscale="Blues",
                            line=dict(color="#25476a", width=1)),
                text=geom.names, textposition="top center",
                textfont=dict(size=8, color="#33475b"),
                hovertext=hover, hoverinfo="text", name="Stations",
            ))
            fig.add_trace(go.Scatter3d(
                x=[train_lon[0]], y=[train_lat[0]], z=[frame_v[0]], mode="markers",
                marker=dict(size=9, color="#d90429",
                            line=dict(color="white", width=1)),
                name="Train",
            ))
            fig.frames = [
                go.Frame(
                    data=[go.Scatter3d(x=[train_lon[i]], y=[train_lat[i]],
                                       z=[frame_v[i]])],
                    traces=[len(fig.data) - 1], name=str(i),
                )
                for i in range(n_frames)
            ]

            _axis = dict(title="", showticklabels=False, showbackground=False,
                         showgrid=False, zeroline=False)
            fig.update_layout(
                height=620, showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0),
                scene=dict(
                    xaxis=_axis, yaxis=_axis,
                    zaxis=dict(title="Speed (km/h)", showbackground=False,
                               gridcolor="rgba(160,175,190,0.25)"),
                    aspectmode="manual", aspectratio=dict(x=1, y=ASPECT, z=0.5),
                    camera=dict(eye=dict(x=1.35, y=-1.35, z=0.9),
                                up=dict(x=0, y=0, z=1)),
                ),
            )

        fig.update_layout(
            updatemenus=[dict(
                type="buttons", showactive=False, x=0.02, y=0.04, xanchor="left",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None, dict(frame=dict(duration=90, redraw=True),
                                          fromcurrent=True, mode="immediate")]),
                    dict(label="Pause", method="animate",
                         args=[[None], dict(frame=dict(duration=0, redraw=False),
                                            mode="immediate")]),
                ],
            )],
            sliders=[dict(
                active=0, x=0.15, len=0.8, y=0.02,
                currentvalue=dict(prefix="t = ", visible=True, xanchor="right"),
                steps=[dict(method="animate", label=fmt_mmss(frame_t[i]),
                            args=[[str(i)], dict(mode="immediate",
                                                 frame=dict(duration=0, redraw=True))])
                       for i in range(n_frames)],
            )],
        )
        chart(fig)
        st.caption(
            "Marker size is published JR East daily boardings, FY2024. The red "
            "section is the only part of the line where segment loading is "
            "measured. Position between stations is interpolated linearly: a "
            "schematic, not a track alignment."
        )

    # ---- schedule --------------------------------------------------------
    with right:
        st.subheader("Circuit schedule")
        st.caption(
            f"Simulated run times plus a {dwell_s:.0f} s dwell at every station. "
            "Departure from Shinagawa at 0:00."
        )
        table = []
        for i, seg in enumerate(segments):
            table.append({
                "Station": seg.to_station,
                "Arrive": fmt_mmss(circuit["arrivals"][i]),
                "Run": f"{circuit['run_times'][i]:.0f}s",
                "km": f"{(geom.cumulative_m[i] + seg.distance_m) / 1000:.1f}",
            })
        st.dataframe(table, use_container_width=True, height=460, hide_index=True)

        st.divider()
        st.markdown(f"**Busiest stations** — published boardings, FY2024")
        top = ridership.nsmallest(5, "rank_2024")[["name_en", "boardings_2024"]]
        for r in top.itertuples():
            st.progress(
                float(r.boardings_2024) / float(ridership["boardings_2024"].max()),
                text=f"{r.name_en} — {r.boardings_2024:,}/day",
            )

    st.divider()
    d1, d2 = st.columns(2)
    with d1:
        st.markdown("##### Rail condition and braking")
        st.markdown(
            f"Adhesion multiplier **{condition.adhesion_multiplier:.2f}** → grip "
            f"ceiling **{condition.max_braking_ms2(0.0) * 3.6:.2f} km/h/s** against a "
            f"**{service_brake_kmhs:.1f} km/h/s** service rate."
        )
        if condition.is_adhesion_limited(service_brake_ms2, 0.0):
            st.error("Grip is the binding constraint — braking is degraded.")
        else:
            st.success("Service brake rate governs; grip is not the constraint.")
        st.caption(
            "Traction is barely affected by rail condition on this class — only "
            "6 of 11 cars are powered but demanded effort is well under even wet-"
            "rail capacity. Braking uses every axle, so that is where grip bites. "
            "Multipliers are modelled, not measured for this line."
        )
    with d2:
        st.markdown("##### What crowding does to the physics")
        base = 1724
        extra_t = (load_factor - 1.0) * base * spec.passenger_mass_kg / 1000
        st.markdown(
            f"At **{crowding_pct}%** the train carries about "
            f"**{int(load_factor * base):,}** passengers, "
            f"{'adding' if extra_t >= 0 else 'removing'} "
            f"**{abs(extra_t):.0f} t** against nominal."
        )
        st.caption(
            "Congestion rate maps onto load factor exactly — both are passengers "
            "over rated capacity — so this published figure feeds the physics "
            "directly. Per-segment loading around the rest of the loop is NOT "
            "public: it would need an origin-destination matrix that JR East does "
            "not release. This model does not invent one."
        )


# ================================================================== ANALYSIS

with view_analysis:
    (tab_seg, tab_trac, tab_loop, tab_energy, tab_coast, tab_val,
     tab_prov, tab_src) = st.tabs(
        ["Segment", "Traction & braking", "The loop", "Energy", "Coasting",
         "Validation", "Provenance", "Sources"]
    )

    # -- Segment -----------------------------------------------------------
    with tab_seg:
        labels = [f"{s.from_station} → {s.to_station} ({s.distance_m:.0f} m)"
                  for s in segments]
        col_a, col_b = st.columns([3, 2])
        with col_a:
            idx = st.selectbox("Segment", range(len(segments)),
                               format_func=lambda i: labels[i], index=0)
        with col_b:
            seg = segments[idx]
            use_coast = st.checkbox("Coast before braking", value=False)
            coast_start = None
            if use_coast:
                coast_start = st.slider("Coast onset (m)", 0.0, float(seg.distance_m),
                                        float(seg.distance_m) * 0.5, 10.0)

        r = run_segment(cls_key, idx, load_factor, power_factor, jerk, brake_kmhs,
                        speed_kmh, dt, coast_start)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Run time", f"{r['run_time_s']:.1f} s")
        m2.metric("Peak speed", f"{r['max_speed_kmh']:.1f} km/h")
        m3.metric("Distance", f"{r['distance_m']:.0f} m")
        m4.metric("Platform error", f"{r['position_error_m']:.2e} m")

        phases = np.array(r["phase"])
        fig = go.Figure()
        for p in Phase:
            mask = phases == p.value
            if not mask.any():
                continue
            fig.add_trace(go.Scatter(
                x=r["x"], y=np.where(mask, r["v"] * 3.6, np.nan), mode="lines",
                name=p.value.capitalize(), line=dict(color=PHASE_COLOUR[p], width=3),
            ))
        fig.add_hline(y=speed_kmh, line_dash="dot", line_color="grey",
                      annotation_text="speed limit")
        fig.update_layout(title=f"{r['label']} — speed against distance",
                          xaxis_title="Distance (m)", yaxis_title="Speed (km/h)",
                          height=420, legend=dict(orientation="h", y=1.12))
        chart(fig)

        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=r["t"], y=r["a"], mode="lines",
                                line=dict(color="#e4572e", width=2)))
        f2.add_hline(y=0, line_color="grey", line_width=1)
        f2.update_layout(title="Acceleration — the sloped edges are the jerk ramps",
                         xaxis_title="Time (s)", yaxis_title="a (m/s²)", height=300)
        chart(f2)

    # -- Traction ----------------------------------------------------------
    with tab_trac:
        st.subheader("Tractive effort — three simultaneous constraints")
        st.caption("Delivered effort is the minimum of all three.")

        v = np.linspace(0.1, spec.max_speed_ms, 400)
        v_kmh = v * 3.6
        adhesion = np.array([
            spec.adhesion_limit_n(vi, load_factor) * condition.adhesion_multiplier
            for vi in v
        ])
        delivered = np.array([traction.effort_n(vi) for vi in v])

        f = go.Figure()
        f.add_trace(go.Scatter(x=v_kmh, y=np.full_like(v, traction.max_effort_n) / 1000,
                               mode="lines", name="Constant effort",
                               line=dict(dash="dash", color="#e4572e")))
        f.add_trace(go.Scatter(x=v_kmh, y=(traction.power_w / v) / 1000, mode="lines",
                               name="Power limited (P/v)",
                               line=dict(dash="dash", color="#17a2b8")))
        f.add_trace(go.Scatter(x=v_kmh, y=adhesion / 1000, mode="lines",
                               name=f"Adhesion ceiling ({condition.name.lower()})",
                               line=dict(dash="dot", color="#7b1fa2")))
        f.add_trace(go.Scatter(x=v_kmh, y=delivered / 1000, mode="lines",
                               name="Delivered", line=dict(color="#212121", width=4)))
        f.add_trace(go.Scatter(x=v_kmh, y=np.array([traction.davis(vi) for vi in v]) / 1000,
                               mode="lines", name="Running resistance",
                               line=dict(color="#4caf50", width=2)))
        f.add_vline(x=traction.base_speed_ms * 3.6, line_dash="dot", line_color="grey",
                    annotation_text="base speed")
        f.update_layout(xaxis_title="Speed (km/h)", yaxis_title="Force (kN)",
                        height=440, yaxis_range=[0, traction.max_effort_n / 1000 * 1.25],
                        legend=dict(orientation="h", y=-0.2))
        chart(f)

        if adhesion.min() < traction.max_effort_n:
            st.warning(
                f"On {condition.name.lower()} rail the traction adhesion ceiling "
                f"falls to {adhesion.min()/1000:.0f} kN, below the "
                f"{traction.max_effort_n/1000:.0f} kN demanded — wheelslip territory."
            )
        else:
            st.success(
                f"Traction adhesion never binds: {traction.max_effort_n/1000:.0f} kN "
                f"demanded against {adhesion.min()/1000:.0f}–{adhesion.max()/1000:.0f} kN "
                f"available on {condition.name.lower()} rail."
            )

        st.divider()
        st.subheader("Jerk-limited braking")
        penalty = brake.a_max / brake.jerk
        b1, b2, b3 = st.columns(3)
        b1.metric("Penalty per stop", f"{penalty:.2f} s")
        b2.metric("Over 30 stops", f"{penalty * 30:.0f} s")
        b3.metric("Of published circuit", f"{100 * penalty * 30 / PUBLISHED_LOOP_S:.1f}%")

        traj = brake.trajectory(speed_kmh / 3.6, dt=0.01)
        fb = go.Figure()
        fb.add_trace(go.Scatter(x=traj["t"], y=traj["v"] * 3.6, mode="lines",
                                name="Speed", line=dict(color="#3f51b5", width=3)))
        fb.add_trace(go.Scatter(x=traj["t"], y=traj["a"], mode="lines", yaxis="y2",
                                name="Deceleration", line=dict(color="#e4572e", width=2)))
        fb.update_layout(title=f"Stop from {speed_kmh:.0f} km/h",
                         xaxis_title="Time (s)", yaxis_title="Speed (km/h)",
                         yaxis2=dict(title="a (m/s²)", overlaying="y", side="right"),
                         height=340, legend=dict(orientation="h", y=1.15))
        chart(fb)

    # -- Loop --------------------------------------------------------------
    with tab_loop:
        rows = run_loop(cls_key, load_factor, power_factor, jerk, brake_kmhs,
                        speed_kmh, dt)
        total = sum(r["run_time_s"] for r in rows)
        reached = sum(1 for r in rows if r["cruise_m"] > 0.5)
        cruise_total = sum(r["cruise_m"] for r in rows)
        dist_total = sum(r["distance_m"] for r in rows)

        l1, l2, l3, l4 = st.columns(4)
        l1.metric("Simulated minimum", f"{total:.0f} s", f"{total/60:.1f} min")
        l2.metric("Published circuit", f"{PUBLISHED_LOOP_S} s",
                  f"{PUBLISHED_LOOP_S - total:+.0f} s", delta_color="off")
        l3.metric("Reaching line speed", f"{reached}/{len(rows)}")
        l4.metric("Distance in cruise", f"{100*cruise_total/dist_total:.0f}%")

        fl = go.Figure()
        fl.add_trace(go.Bar(
            x=[r["label"] for r in rows], y=[r["run_time_s"] for r in rows],
            marker_color=["#17a2b8" if r["cruise_m"] > 0.5 else "#e4572e" for r in rows],
        ))
        fl.update_layout(title="Run time per segment (blue = reaches line speed)",
                         yaxis_title="Seconds", height=430, xaxis_tickangle=-60)
        chart(fl)

    # -- Energy ------------------------------------------------------------
    with tab_energy:
        e = run_energy(cls_key, load_factor, power_factor, jerk, brake_kmhs,
                       speed_kmh, dt, dwell_s, receptivity, aux_kw)

        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Net per circuit", f"{e['net_kwh']:.0f} kWh")
        e2.metric("Per car-km", f"{e['kwh_per_car_km']:.2f} kWh")
        e3.metric("Regen saving", f"{100*e['regen_saving']:.0f}%", "of gross",
                  delta_color="off")
        e4.metric("Per passenger-km", f"{e['wh_per_pax_km']:.1f} Wh")

        st.caption(
            "Published heavy-metro consumption is typically 2–4 kWh per car-km. "
            "That is a range rather than a measurement of this line, so it is a "
            "weak check — but a figure outside it would be a clear warning."
        )

        st.subheader("Where the energy goes")
        wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "total"],
            x=["Traction work", "Auxiliary load", "Regenerated", "Net from supply"],
            y=[e["traction_kwh"], e["aux_kwh"], -e["regen_kwh"], 0],
            text=[f"{e['traction_kwh']:.0f}", f"+{e['aux_kwh']:.0f}",
                  f"−{e['regen_kwh']:.0f}", f"{e['net_kwh']:.0f}"],
            textposition="outside",
            connector=dict(line=dict(color="#9e9e9e")),
            increasing=dict(marker=dict(color="#e4572e")),
            decreasing=dict(marker=dict(color="#4caf50")),
            totals=dict(marker=dict(color="#212121")),
        ))
        wf.update_layout(yaxis_title="kWh per circuit", height=400)
        chart(wf)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Traction work, by destination")
            share = go.Figure(go.Bar(
                x=["Running resistance", "Absorbed by brakes"],
                y=[e["resistance_kwh"], e["brake_kwh"]],
                marker_color=["#4caf50", "#e4572e"],
                text=[f"{100*e['resistance_kwh']/e['traction_kwh']:.0f}%",
                      f"{100*e['brake_kwh']/e['traction_kwh']:.0f}%"],
                textposition="outside",
            ))
            share.update_layout(yaxis_title="kWh", height=330, showlegend=False)
            chart(share)
            st.caption(
                f"{100*e['brake_kwh']/e['traction_kwh']:.0f}% is dissipated in "
                "braking rather than against resistance: 30 stops in 34.5 km "
                "means accelerating 480 t to line speed and discarding the "
                "kinetic energy each time. Regeneration therefore recovers more "
                "here than on a long-distance line, where the energy goes to "
                "aerodynamic drag."
            )
        with c2:
            st.markdown("##### Cumulative around the loop")
            cum = go.Figure()
            cum.add_trace(go.Scatter(x=e["profile_km"], y=e["profile_traction"],
                                     mode="lines", name="Traction (gross)",
                                     line=dict(color="#e4572e", width=3)))
            cum.add_trace(go.Scatter(x=e["profile_km"], y=e["profile_net"],
                                     mode="lines", name="Net of regeneration",
                                     line=dict(color="#3f51b5", width=3)))
            cum.update_layout(xaxis_title="Distance around the loop (km)",
                              yaxis_title="kWh", height=330,
                              legend=dict(orientation="h", y=1.15))
            chart(cum)
            st.caption(
                "Each riser is one acceleration; each flat section is a coast or "
                "brake."
            )

        st.subheader("Energy per segment")
        seg_fig = go.Figure(go.Bar(
            x=e["seg_labels"], y=e["seg_net"], marker_color="#3f51b5",
            customdata=[[d, t] for d, t in zip(e["seg_distance"], e["seg_traction"])],
            hovertemplate="%{x}<br>net %{y:.1f} kWh<br>traction "
                          "%{customdata[1]:.1f} kWh<br>%{customdata[0]:.0f} m"
                          "<extra></extra>",
        ))
        seg_fig.update_layout(yaxis_title="Net kWh", height=400, xaxis_tickangle=-60)
        chart(seg_fig)

        st.divider()
        st.markdown("##### Energy balance check")
        st.markdown(
            "Each segment starts and ends at rest, so the kinetic term vanishes "
            "and E_traction = E_resistance + E_brake_system must hold exactly. "
            "Both sides are computed independently, which tests the integrator "
            "rather than restating an assumption."
        )
        err = e["balance_error"]
        if err < 5e-3:
            st.success(f"Balance closes to {err:.2e} relative error at dt = {dt} s.")
        else:
            st.error(
                f"Balance error {err:.2e} is too large to be discretisation. "
                "Something is wrong in the accounting — reduce the integration "
                "step to check whether it converges."
            )
        st.caption(
            "This check previously failed at 7.3%, caused by cruise force being "
            "recorded as effort_n, which returns zero at maximum speed — where "
            "cruise occurs on this line. Run times were unaffected; energy "
            "figures were wrong until corrected."
        )

    # -- Sources -----------------------------------------------------------
    with tab_src:
        st.subheader("Sources")

        st.markdown("##### Line and timetable")
        st.table([
            {"Quantity": "Station sequence, operating distance (営業キロ), 34.5 km total",
             "Source": "Yamanote Line, English Wikipedia",
             "Link": "https://en.wikipedia.org/wiki/Yamanote_Line"},
            {"Quantity": "Maximum gradient 34‰ (Tabata–Nishi-Nippori), line speed 90 km/h",
             "Source": "山手線, Japanese Wikipedia",
             "Link": "https://ja.wikipedia.org/wiki/山手線"},
            {"Quantity": "Circuit times: 3,948 s inner mean, 3,937 s outer, 3,840 s fastest, 4,200 s slowest",
             "Source": "Published timetable aggregates, secondary compilation",
             "Link": "https://otoku-info100.com/yamate-round-time241015/"},
            {"Quantity": "Per-segment scheduled times",
             "Source": "NOT COMMITTED — sources disagreed and appeared column-shifted",
             "Link": "—"},
        ])

        st.markdown("##### Rolling stock")
        st.table([
            {"Class": "E235-0 — 6M5T, 340.8 t, 1,724 capacity, MT79 140 kW ×24, "
                      "3.0 / 4.2 km/h/s, 90 km/h",
             "Source": "JR東日本E235系電車, Japanese Wikipedia",
             "Link": "https://ja.wikipedia.org/wiki/JR東日本E235系電車"},
            {"Class": "E233-0 — 6M4T, 318.8 t, 1,582 capacity, MT75 140 kW, 100 km/h",
             "Source": "JR東日本E233系電車, Japanese Wikipedia",
             "Link": "https://ja.wikipedia.org/wiki/JR東日本E233系電車"},
            {"Class": "E5 — 8M2T, 453.5 t, 9,600 kW, 320 km/h, 1.71 km/h/s",
             "Source": "新幹線E5系電車, Japanese Wikipedia",
             "Link": "https://ja.wikipedia.org/wiki/新幹線E5系電車"},
            {"Class": "E7/W7 — 10M2T, 540 t, 12,000 kW, speed-dependent braking curve",
             "Source": "新幹線E7系・W7系電車, Japanese Wikipedia",
             "Link": "https://ja.wikipedia.org/wiki/新幹線E7系・W7系電車"},
            {"Class": "N700S — 14M2T, mass bounded < 700 t, 17,080 kW, 2.6 km/h/s",
             "Source": "新幹線N700S系電車, Japanese Wikipedia",
             "Link": "https://ja.wikipedia.org/wiki/新幹線N700S系電車"},
        ])

        st.markdown("##### Operations and demand")
        st.table([
            {"Quantity": "Recovery margin and dwell practice, all major Japanese operators",
             "Source": "MLIT 事故防止に係る総点検の実施結果（運行計画）, PDF",
             "Link": "https://www.mlit.go.jp/kisha/kisha05/08/080722_3/01.pdf"},
            {"Quantity": "Station boardings FY2024, all 30 stations",
             "Source": "JR East 各駅の乗車人員 (primary blocks automated access; "
                       "figures via secondary compilation)",
             "Link": "https://www.jreast.co.jp/company/data/passenger/"},
            {"Quantity": "Peak congestion 125%, Ueno→Okachimachi, 07:43–08:43",
             "Source": "MLIT 都市鉄道の混雑率調査結果 FY2023, via secondary report",
             "Link": "https://ueno.keizai.biz/headline/752/"},
            {"Quantity": "Regeneration rate 59.0% (E235, Yamanote); 47% (E231)",
             "Source": "JR East, via secondary summary — denominator unverified",
             "Link": "https://www.jreast.co.jp/press/2024/20240508_ho01.pdf"},
            {"Quantity": "Optimal-driving energy saving 12% and 15.7%, Yamanote",
             "Source": "JSME 鉄道分野における省エネ技術の研究開発",
             "Link": "https://www.jsme.or.jp/kaisi/1240-13/"},
        ])

        st.markdown("##### Fetched programmatically")
        st.table([
            {"Data": "Station coordinates, 30 stations (display only)",
             "Service": "OpenStreetMap via Overpass API",
             "Link": "https://overpass-api.de/api/interpreter"},
            {"Data": "Station ground elevations, for the gradient bounding study",
             "Service": "Open-Elevation",
             "Link": "https://api.open-elevation.com/api/v1/lookup"},
            {"Data": "Real-time train positions (Toei only; no JR East)",
             "Service": "ODPT public mirror, no key required",
             "Link": "https://api-public.odpt.org/api/v4/odpt:Train"},
            {"Data": "Real-time train positions, JR East — requires acl:consumerKey",
             "Service": "ODPT authenticated API",
             "Link": "https://api.odpt.org/api/v4/odpt:Train"},
        ])

        st.divider()
        st.markdown("##### Equations")

        eqs = [
            ("Equation of motion",
             r"m_{\mathrm{eff}}\,\frac{dv}{dt} = F_{\mathrm{traction}}(v)"
             r" - R_{\mathrm{run}}(v) - R_{\mathrm{grade}} - R_{\mathrm{curve}}",
             "segment.py, traction.py"),
            ("Effective mass (rotating inertia)",
             r"m_{\mathrm{eff}} = m\,(1 + \lambda),\qquad \lambda \approx 0.08",
             "stock.py"),
            ("Davis running resistance",
             r"R(v) = A + Bv + Cv^{2}", "resistance.py"),
            ("Aerodynamic coefficient",
             r"C = \tfrac{1}{2}\,\rho\,\big(C_{d,\mathrm{ends}} + "
             r"C_{d,\mathrm{car}}\,n\big)\,A_f", "resistance.py"),
            ("Grade resistance (per-mille)",
             r"R_{\mathrm{grade}} = m g \sin\theta \approx m g \cdot "
             r"\frac{\text{grade}_{‰}}{1000}", "resistance.py"),
            ("Curve resistance",
             r"R_{\mathrm{curve}} = \frac{k\,m}{R}", "resistance.py"),
            ("Tractive effort, three constraints",
             r"F(v) = \min\Big(F_{\max},\ \frac{P}{v},\ "
             r"\mu(v)\,m_{\mathrm{adh}}\,g\Big)", "traction.py"),
            ("Starting effort from published acceleration",
             r"F_{\max} = m_{\mathrm{eff}}\,a_{\mathrm{start}} + R(0)",
             "stock.py"),
            ("Base speed",
             r"v_{\mathrm{base}} = P / F_{\max}", "traction.py"),
            ("Adhesion (Curtius–Kniffler)",
             r"\mu(v) = 0.161 + \frac{7.5}{v_{\mathrm{km/h}} + 44}", "stock.py"),
            ("Jerk-limited stop, trapezoidal time",
             r"T = \frac{v_0}{a_{\max}} + \frac{a_{\max}}{j}", "brake.py"),
            ("Trapezoidal/triangular threshold",
             r"v_0 \ge \frac{a_{\max}^{2}}{j}", "brake.py"),
            ("Brake application point",
             r"x + d_{\mathrm{stop}}(v) \ge L", "segment.py"),
            ("Traction energy",
             r"E = \int F_{\mathrm{traction}}\,v\;dt", "energy.py"),
            ("Energy balance identity (start and end at rest)",
             r"E_{\mathrm{traction}} = E_{\mathrm{resistance}} + "
             r"E_{\mathrm{brake}}", "energy.py"),
            ("Recovered energy",
             r"E_{\mathrm{regen}} = \eta\,\rho\,\!\!\int_{v>v_{\mathrm{cut}}}"
             r"\!\! P_{\mathrm{brake}}\;dt", "energy.py"),
            ("Circuit identity",
             r"T_{\mathrm{circuit}} = \textstyle\sum T_{\mathrm{run}} + "
             r"n\,T_{\mathrm{dwell}} + T_{\mathrm{margin}}", "validate.py"),
            ("Inverted inference for dwell",
             r"T_{\mathrm{dwell}} = \frac{T_{\mathrm{published}} - "
             r"T_{\mathrm{std}}(1 + \text{margin})}{n}", "validate.py"),
        ]
        for name, tex, where in eqs:
            c1, c2 = st.columns([3, 1])
            with c1:
                st.latex(tex)
            with c2:
                st.markdown(f"**{name}**<br><span style='color:#5B6B7A;"
                            f"font-size:0.75rem'>{where}</span>",
                            unsafe_allow_html=True)

        st.caption(
            "Integration is fixed-step RK4 with explicit control-phase "
            "switching. Root finding is bisection. scipy is not a dependency: "
            "the braking solution is closed-form and the brake application "
            "point is bracketed, so neither required it."
        )

        st.divider()
        st.markdown("##### Dependencies between quantities")
        st.table([
            {"Change": "Load factor up", "Effect": "Mass up, so acceleration down, "
             "traction energy up; braking rate unchanged (published figure is net)"},
            {"Change": "Speed up", "Effect": "Aerodynamic resistance rises with v²; "
             "above base speed available effort falls as P/v"},
            {"Change": "Jerk limit down", "Effect": "Stopping time rises by a_max/j "
             "per stop, independent of initial speed"},
            {"Change": "Rail condition degraded", "Effect": "Adhesion ceiling falls; "
             "binds on braking only in the leaf-fall case, never on traction"},
            {"Change": "Regeneration receptivity up", "Effect": "Net energy down; "
             "advantage of coasting down, since the two are partial substitutes"},
            {"Change": "Coast onset later", "Effect": "Run time down, energy up — "
             "the trade-off traced by the Pareto front"},
            {"Change": "Assumed dwell up", "Effect": "Implied recovery margin down; "
             "one equation in two unknowns, so they cannot both be free"},
            {"Change": "Gradient introduced", "Effect": "Gravitational work cancels "
             "over a closed loop; run time and energy still rise in both "
             "directions, because climbing is power-limited"},
        ])

        st.divider()
        st.markdown("##### Code")
        st.table([
            {"Module": "units.py", "Contains": "km/h/s ↔ m/s² conversion; all unit "
             "handling in one place"},
            {"Module": "stock.py", "Contains": "TrainSpec, effective mass, adhesion "
             "coefficient, starting effort"},
            {"Module": "resistance.py", "Contains": "Davis, grade, curve resistance; "
             "coefficient estimator"},
            {"Module": "traction.py", "Contains": "Effort curve, capability vs "
             "applied force, holding force"},
            {"Module": "brake.py", "Contains": "Jerk-limited braking, closed form"},
            {"Module": "segment.py", "Contains": "RK4 integrator, control-phase "
             "switching, brake-point bisection"},
            {"Module": "network.py", "Contains": "Loop geometry, whole-circuit "
             "simulation"},
            {"Module": "energy.py", "Contains": "Traction energy, losses, "
             "regeneration, balance check"},
            {"Module": "coasting.py", "Contains": "Four-phase solver, Pareto front"},
            {"Module": "fleet.py", "Contains": "Cross-class comparison, bounded "
             "quantities"},
            {"Module": "conditions.py", "Contains": "Rail surface condition, braking "
             "adhesion ceiling"},
            {"Module": "validate.py", "Contains": "Decomposition ladder, "
             "identifiability locus, inverted inference"},
            {"Module": "odpt.py", "Contains": "Real-time client, censoring-aware "
             "dwell measurement (unused without a key)"},
        ])

        st.divider()
        st.markdown("##### Access limitations")
        st.markdown(
            "- JR East returns HTTP 403 to automated requests for its own "
            "ridership page and press PDFs. Those figures reach this project "
            "through secondary compilations, and the 回生率 denominator could "
            "not be confirmed at source.\n"
            "- ODPT real-time data for JR East requires a registered consumer "
            "key. The unauthenticated mirror carries Toei only.\n"
            "- vocab.odpt.org serves an expired certificate; the odpt:Train "
            "field semantics were confirmed against the live feed instead.\n"
            "- Per-segment scheduled times are published only to the minute, so "
            "±30 s exceeds the effect being measured and no per-segment "
            "timetable is committed.\n"
            "- The Tokyo Subway Route Map (Bureau of Transportation TMG / Tokyo "
            "Metro) was supplied as a visual reference for the 2D diagram only; "
            "no data was taken from it."
        )

    # -- Coasting ----------------------------------------------------------
    with tab_coast:
        st.subheader("Energy against run time")
        st.markdown(
            "The energy-minimal profile for a fixed distance and time is "
            "accelerate, cruise, coast, brake. With the shape fixed the strategy "
            "reduces to one parameter, the coast onset, found by bisection on "
            "run time."
        )

        ci = st.selectbox("Segment", range(len(segments)),
                          format_func=lambda i: (
                              f"{segments[i].from_station} → {segments[i].to_station} "
                              f"({segments[i].distance_m:.0f} m)"),
                          index=int(np.argmax([s.distance_m for s in segments])),
                          key="coast_seg")

        with st.spinner("Solving coast onset for each target time…"):
            rows = run_pareto(cls_key, ci, load_factor, power_factor, jerk,
                              brake_kmhs, speed_kmh, receptivity, aux_kw)

        if not rows:
            st.warning("No achievable targets for this segment and configuration.")
        else:
            base = rows[0]
            slowest = rows[-1]
            p1, p2, p3 = st.columns(3)
            p1.metric("Flat out", f"{base['coast_kwh']:.1f} kWh",
                      f"{base['time_s']:.0f} s", delta_color="off")
            p2.metric(f"At {slowest['factor']:.0%} time",
                      f"{slowest['coast_kwh']:.1f} kWh",
                      f"{100*(slowest['coast_kwh']/base['coast_kwh']-1):.0f}% energy",
                      delta_color="off")
            p3.metric("Coasting vs slower cruise",
                      f"{100*slowest['saving']:+.1f}%", "at equal run time",
                      delta_color="off")

            pf = go.Figure()
            pf.add_trace(go.Scatter(
                x=[r["time_s"] for r in rows], y=[r["coast_kwh"] for r in rows],
                mode="lines+markers", name="Coast (four-phase)",
                line=dict(color="#3f51b5", width=3), marker=dict(size=9),
            ))
            pf.add_trace(go.Scatter(
                x=[r["time_s"] for r in rows], y=[r["slower_kwh"] for r in rows],
                mode="lines+markers", name="Cruise slower (naive)",
                line=dict(color="#e4572e", width=3, dash="dash"), marker=dict(size=9),
            ))
            pf.update_layout(xaxis_title="Run time (s)", yaxis_title="Net energy (kWh)",
                             height=430, legend=dict(orientation="h", y=1.13))
            chart(pf)

            st.markdown(
                f"Coasting beats cruising slower by {100*slowest['saving']:.1f}% "
                f"at equal run time. Allowing {slowest['factor']-1:.0%} more "
                f"time reduces energy by "
                f"{100*(1-slowest['coast_kwh']/base['coast_kwh']):.0f}%, since "
                "kinetic energy scales with the square of speed."
            )

            st.markdown("##### Where coasting starts")
            cf = go.Figure()
            cf.add_trace(go.Bar(
                x=[f"{r['factor']:.0%}" for r in rows[1:]],
                y=[r["coast_distance_m"] for r in rows[1:]],
                marker_color="#f5b700",
                text=[f"{r['coast_start_m']:.0f} m in" for r in rows[1:]],
                textposition="outside",
            ))
            cf.update_layout(xaxis_title="Run time, as a multiple of the minimum",
                             yaxis_title="Distance spent coasting (m)", height=330)
            chart(cf)
            st.caption(
                "Coasting and regeneration are partial substitutes. Raising "
                "receptivity reduces the advantage of coasting, so the value of "
                "this optimisation depends on an unpublished parameter."
            )

    # -- Validation --------------------------------------------------------
    with tab_val:
        st.subheader("From theoretical minimum to published timetable")
        v1, v2 = st.columns(2)
        with v1:
            derate = st.slider("Run curve de-rating (km/h below limit)", 0.0, 5.0, 3.0, 0.5)
        with v2:
            round_up = st.slider("Round-up quantum (s)", 1.0, 10.0, 5.0, 1.0)

        d = run_decompose(cls_key, load_factor, power_factor, jerk, brake_kmhs,
                          speed_kmh, dwell_s, derate, round_up, PUBLISHED_LOOP_S)
        names = [n for n, _, _ in d["rungs"]]
        deltas = [dl for _, _, dl in d["rungs"]]

        fw = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(names) - 1),
            x=names, y=deltas, text=[f"{x:+.0f} s" for x in deltas],
            textposition="outside", connector=dict(line=dict(color="#9e9e9e")),
            increasing=dict(marker=dict(color="#17a2b8")),
        ))
        fw.add_hline(y=PUBLISHED_LOOP_S, line_dash="dot", line_color="#e4572e",
                     annotation_text=f"published {PUBLISHED_LOOP_S} s")
        fw.update_layout(title="Decomposition ladder — each layer is documented practice",
                         yaxis_title="Cumulative circuit time (s)", height=440,
                         xaxis_tickangle=-20)
        chart(fw)

        r1, r2, r3 = st.columns(3)
        r1.metric("Standard run time", f"{d['standard_run_time_s']:.0f} s")
        r2.metric("Total dwell", f"{dwell_s * len(segments):.0f} s")
        r3.metric("Residual margin", f"{d['residual_margin_s']:.0f} s",
                  f"{100*d['residual_margin_fraction']:.1f}%", delta_color="off")

        if d["residual_margin_fraction"] > 0.12:
            st.warning(
                f"{100*d['residual_margin_fraction']:.0f}% is far above documented "
                "practice. Either dwell is higher than assumed, or the model's "
                "minimum is too fast — most plausibly because signal and ATC "
                "restrictions are not modelled."
            )
        elif d["residual_margin_fraction"] < 0:
            st.error("Negative residual: no room for any margin, which contradicts "
                     "documented JR East practice.")
        else:
            st.success(f"{100*d['residual_margin_fraction']:.1f}% is consistent with "
                       "documented recovery-margin practice.")

        st.divider()
        st.subheader("The identifiability problem")
        st.markdown(
            "Circuit time = run time + margin + dwell is one equation in two "
            "unknowns. Every point on this line fits the published circuit time; "
            "public data cannot distinguish between them."
        )
        locus = identifiability_locus(d["standard_run_time_s"], PUBLISHED_LOOP_S,
                                      len(segments), (15.0, 55.0), 81)
        fi = go.Figure()
        fi.add_trace(go.Scatter(x=locus["dwell_s"], y=100 * locus["margin_fraction"],
                                mode="lines", line=dict(color="#212121", width=3),
                                name="consistent with published circuit"))
        fi.add_hrect(y0=0, y1=10, fillcolor="#4caf50", opacity=0.15, line_width=0,
                     annotation_text="documented practice")
        fi.add_trace(go.Scatter(x=[dwell_s], y=[100 * d["residual_margin_fraction"]],
                                mode="markers", marker=dict(size=15, color="#e4572e"),
                                name="current assumption"))
        fi.update_layout(xaxis_title="Mean station dwell (s)",
                         yaxis_title="Implied recovery margin (%)", height=400,
                         legend=dict(orientation="h", y=1.12))
        chart(fi)

        zero = dwell_implied_by_zero_margin(d["standard_run_time_s"], PUBLISHED_LOOP_S,
                                            len(segments))
        st.markdown(f"Dwell above {zero:.1f} s would imply no recovery margin, "
                    "which documented JR East practice excludes. This bound "
                    "requires no modelling assumption.")

        st.divider()
        st.subheader("Inverted inference — the actual result")
        st.markdown(
            "Margin practice is documented; dwell is not. Fixing margin at "
            "documented levels and solving for dwell runs the inference in the "
            "direction the evidence supports."
        )
        margins = [0.03, 0.05, 0.08, 0.10]
        implied = [dwell_implied_by_margin(d["standard_run_time_s"], PUBLISHED_LOOP_S,
                                           len(segments), m) for m in margins]
        fii = go.Figure(go.Bar(x=[f"{100*m:.0f}%" for m in margins], y=implied,
                               marker_color="#3f51b5",
                               text=[f"{x:.1f} s" for x in implied],
                               textposition="outside"))
        fii.update_layout(title="Implied mean station dwell",
                          xaxis_title="Assumed recovery margin",
                          yaxis_title="Implied dwell (s)", height=340)
        chart(fii)
        st.caption(
            f"Implied dwell spans {min(implied):.1f}–{max(implied):.1f} s. The "
            "estimate is well-conditioned because physics error is multiplied by "
            f"run time ({d['standard_run_time_s']:.0f} s) while dwell is "
            "multiplied by 30 stations."
        )

    # -- Provenance --------------------------------------------------------
    with tab_prov:
        st.subheader("Parameter provenance")
        st.markdown(
            "Each parameter is tagged at source and the tag is carried through "
            "the code, so no modelled value is presented as a measured one."
        )

        badge = {
            "published": "published",
            "derived": "derived",
            "modelled": "modelled",
            "published_ambiguous": "published, ambiguous",
            "bounded": "bounded only",
            "unknown": "unknown",
            "not_modelled": "not modelled",
        }

        st.markdown("##### Rolling stock and line")
        st.table([
            {"Parameter": "Station kilometrage (営業キロ)", "Status": badge["published"],
             "Source / basis": "JR East operating distance; sums to 34.5 km"},
            {"Parameter": "Formation, mass, capacity", "Status": badge["published"],
             "Source / basis": "6M5T, 340.8 t, 1,724 — ja.wikipedia E235系"},
            {"Parameter": "Traction power", "Status": badge["derived"],
             "Source / basis": "3,360 kW = 6 cars × 4 motors × 140 kW (one-hour rating)"},
            {"Parameter": "Acceleration / deceleration", "Status": badge["published"],
             "Source / basis": "3.0 and 4.2 km/h/s"},
            {"Parameter": "Davis coefficients", "Status": badge["modelled"],
             "Source / basis": "No published set for this class. Estimated from "
                               "mass, car count, frontal area. Largest uncertainty."},
            {"Parameter": "Rotational inertia λ", "Status": badge["modelled"],
             "Source / basis": "0.08, midpoint of the 0.06–0.10 EMU range"},
            {"Parameter": "Comfort jerk limit", "Status": badge["modelled"],
             "Source / basis": "0.75 m/s³, midpoint of literature range"},
            {"Parameter": "Gradient", "Status": badge["not_modelled"],
             "Source / basis": "Evaluated at zero. Bounded separately: worst "
                               "segment ±7.7 s, circuit only 0.46%"},
            {"Parameter": "Signal / ATC restrictions", "Status": badge["unknown"],
             "Source / basis": "Not public. Largest unmodelled contributor to run time."},
        ])

        st.markdown("##### Operations and demand")
        st.table([
            {"Parameter": "Circuit times", "Status": badge["published"],
             "Source / basis": "3,948 s inner mean; 3,840 s fastest"},
            {"Parameter": "Operator margin practice", "Status": badge["published"],
             "Source / basis": "MLIT 運行計画 survey — de-rating, round-up, added margin"},
            {"Parameter": "Station boardings", "Status": badge["published"],
             "Source / basis": "JR East 各駅の乗車人員, FY2024, all 30 stations"},
            {"Parameter": "Peak congestion", "Status": badge["published"],
             "Source / basis": "MLIT 125%, Ueno→Okachimachi, one section only"},
            {"Parameter": "Segment loading elsewhere", "Status": badge["unknown"],
             "Source / basis": "Needs an OD matrix. Not public. Not invented here."},
            {"Parameter": "Mean dwell", "Status": badge["unknown"],
             "Source / basis": "Never published. Inferred 39–50 s from margin practice."},
            {"Parameter": "Per-segment scheduled times", "Status": badge["published_ambiguous"],
             "Source / basis": "Minute-resolution only; ±30 s swamps the effect"},
            {"Parameter": "Regen receptivity", "Status": badge["modelled"],
             "Source / basis": "0.70 assumed. JR East 回生率 of 59.0% implies ~0.76, "
                               "but the ratio's denominator is unverified."},
            {"Parameter": "N700S formation mass", "Status": badge["bounded"],
             "Source / basis": "Published only as 'under 700 t' — carried as a bound"},
        ])

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Stated assumptions")
            st.markdown(
                "- Flat gradient (known false on one segment; bounded)\n"
                "- No signal or ATC restrictions\n"
                "- No per-segment curve or turnout limits\n"
                "- Dry rail, no wind\n"
                "- Full tractive effort throughout acceleration\n"
                "- Published deceleration treated as a *net* rate\n"
                "- One-hour power rating, not short-term overload — this makes "
                "the model **slower** than reality, so any measured fast bias is "
                "a lower bound"
            )
        with c2:
            st.markdown("##### Where it would be wrong")
            st.markdown(
                "- If true run time is materially higher than modelled — most "
                "plausibly through unmodelled ATC — implied dwell falls "
                "proportionally. A 17% error moves it from ~45 s to ~34 s.\n"
                "- Per-segment claims are unsafe: minute-rounded schedules and "
                "unmodelled gradient both bite at that scale.\n"
                "- Shinkansen aerodynamics do **not** follow from the commuter "
                "Davis estimator, which is why those classes are refused rather "
                "than rendered."
            )

        st.divider()
        st.markdown("##### Numerical checks")
        n1, n2, n3 = st.columns(3)
        n1.metric("Platform position error", "0.0 m", "all 30 segments",
                  delta_color="off")
        n2.metric("Circuit time convergence", "8 ms", "dt 0.5 → 0.005 s",
                  delta_color="off")
        n3.metric("Energy balance residual", "2.5e-04", "first order in dt",
                  delta_color="off")
        st.caption(
            "The energy balance is a real test, not a restatement: a segment "
            "starts and ends at rest, so E_traction = E_resistance + E_brake "
            "must hold, and both sides are computed independently. It initially "
            "failed at 7.3% and exposed a genuine bug — cruise force was being "
            "recorded as zero because `effort_n` returns zero at maximum speed, "
            "which is exactly where cruise happens on this line."
        )
