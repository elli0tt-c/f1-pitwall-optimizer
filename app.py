"""
app.py
F1 / Endurance Pit-Wall Strategy Simulator (v2.4 — 2026 Season FastF1, Tooltips & Engineering Manual Edition)

Interactive Streamlit + Plotly pit-wall command center featuring:
- All 23 circuits of the 2026 FIA Formula 1 World Championship calibrated via OpenF1 & FastF1
- Exact Backward Dynamic Programming (Bellman optimality) solver enforcing the 2-compound rule
- Interactive Custom Pit-Wall Strategy Builder with FIA 2-compound rule validation & DP delta benchmarking
- Real Grand Prix Result Strategy loader (P1–P20 actual driver pit strategies vs. DP optimum)
- Live mid-race chaos controls (Safety Car deployment, Rival Undercut, Passing Rain Cell)
- Monte Carlo stochastic race rollout engine with P10–P90 confidence fan chart
- Circuit-specific empirical tire compound degradation & thermal cliff curves
- Native '?' tooltips on every metric/control + full Engineering, Physics & Rules Manual view
"""

import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine.optimizer import (
    build_actual_driver_plan,
    build_custom_strategy_plan,
    generate_standard_rival_strategies,
    load_actual_strategies,
    run_monte_carlo_evaluation,
    solve_optimal_strategy_dp,
)
from engine.physics import (
    COMPOUNDS,
    DRY_COMPOUNDS,
    CarStrategyPlan,
    compute_tire_delta,
    load_circuits,
    simulate_race_trace,
)

st.set_page_config(
    page_title="F1 Pit-Wall Strategy Optimizer | 2026 FastF1 Edition",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dark pit-wall styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    div[data-testid="stMetric"] {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 14px 18px;
    }
    .telemetry-badge {
        display: inline-block;
        background-color: #161B22;
        border: 1px solid #2EA043;
        color: #3FB950;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .manual-card {
        background-color: #161B22;
        border: 1px solid #30363D;
        border-radius: 10px;
        padding: 18px 22px;
        margin-bottom: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _file_mtime(path: str) -> float:
    return os.path.getmtime(path) if os.path.exists(path) else 0.0


@st.cache_data
def get_all_circuits(_mtime: float):
    return load_circuits()


@st.cache_data
def get_all_actual_strategies(_mtime: float):
    return load_actual_strategies()


def render_engineering_manual(circuit) -> None:
    """Renders the complete Physics, FIA Sporting Rules, Bellman DP & Telemetry Manual."""
    st.title("📘 Engineering, Physics & FIA Rules Manual")
    st.markdown(
        '<div class="telemetry-badge">📐 Mathematical & Sporting Specification &nbsp;|&nbsp; '
        'F1 / Endurance Pit-Wall Strategy Simulator (v2.4)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "This manual documents every physical equation, FIA sporting constraint, Bellman Dynamic Programming "
        "recursion, Monte Carlo stochastic model, and OpenF1/FastF1 telemetry calibration pipeline powering "
        "this simulator. You can also hover over the **`?`** icon on any metric or slider in the **Live Pit-Wall "
        "Simulator** for instant context."
    )

    tab_physics, tab_rules, tab_dp, tab_mc, tab_etl, tab_glossary = st.tabs(
        [
            "1. Lap-Time Physics",
            "2. FIA Sporting Rules",
            "3. Bellman DP Math",
            "4. Monte Carlo & Chaos",
            "5. Telemetry ETL Pipeline",
            "6. Metrics Glossary",
        ]
    )

    # --- TAB 1: LAP-TIME PHYSICS ---
    with tab_physics:
        st.header("1. Deterministic Lap-Time Physics Engine (`engine/physics.py`)")
        st.markdown(
            "Every simulated race lap $l \\in \\{1, \\dots, N\\}$ computes instantaneous lap time $T_{\\text{lap}}(l)$ "
            "as an additive composition of calibrated circuit baseline pace, fuel burn-off mass reduction, "
            "non-linear tire degradation, track wetness crossover physics, dirty-air traffic penalties, and "
            "Safety Car / pit-lane time deltas:"
        )
        st.latex(
            r"T_{\text{lap}}(l) = T_{\text{base}} + \Delta T_{\text{fuel}}(l) + \Delta T_{\text{tire}}(c, a, w_l) "
            r"+ \Delta T_{\text{traffic}}(l) + \Delta T_{\text{driver}} + \Delta T_{\text{pit}}(l)"
        )

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("A. Fuel Mass Burn-Off Effect")
            st.markdown(
                "Cars start a Grand Prix with roughly **105–110 kg** of fuel and burn it linearly across the race. "
                "Lighter cars accelerate faster and brake later, yielding a linear lap-time improvement:"
            )
            st.latex(
                r"\Delta T_{\text{fuel}}(l) = - k_{\text{fuel}} \cdot \frac{l - 1}{N - 1}"
            )
            st.markdown(
                "- **$T_{\\text{base}}$**: Calibrated clean-air race baseline lap time (s) for the selected circuit.\n"
                "- **$k_{\\text{fuel}}$**: Total fuel-burn lap-time gain from Lap 1 heavy fuel to final-lap light fuel "
                "(typically **1.6s – 2.2s** across a Grand Prix distance)."
            )

            st.subheader("B. Dirty Air & Overtaking Resistance")
            st.markdown(
                "When the player car closes within aerodynamic wake range (less than ~1.5s behind a rival) without "
                "completing a pass, turbulent airflow reduces front downforce and costs **0.25s – 0.45s/lap** "
                "depending on circuit overtaking difficulty—creating the classic **undercut** incentive to box early "
                "into clean air."
            )

        with c2:
            st.subheader("C. Empirical Tire Wear & Thermal Cliff")
            st.markdown(
                "Each tire compound $c \\in \\{\\text{SOFT}, \\text{MEDIUM}, \\text{HARD}, \\text{INTER}, \\text{WET}\\}$ "
                "has a fresh-rubber pace offset $\\Delta T_{0,c}$, a linear wear slope $m_c$, and a **thermal cliff age** "
                "$A_{\\text{cliff}, c}$ beyond which rubber grain/blistering accelerates quadratically:"
            )
            st.latex(
                r"\Delta T_{\text{tire}}(c, a) = \Delta T_{0,c} + m_c \cdot a + "
                r"\gamma_{\text{cliff}} \cdot \max(0, a - A_{\text{cliff}, c})^2"
            )
            st.markdown(
                "- **Soft (C5/C4)**: Fastest initial grip ($\\Delta T_{0} = 0.00\\text{s}$), steepest linear wear, earliest thermal cliff.\n"
                "- **Medium (C3)**: Balanced race tire ($\\Delta T_{0} \\approx +0.45\\text{s}$), moderate degradation.\n"
                "- **Hard (C2/C1)**: Durable endurance compound ($\\Delta T_{0} \\approx +0.90\\text{s}$), flat wear curve and late cliff."
            )

        st.subheader("D. Wet-Track Crossover & Compound Mismatch Penalty")
        st.markdown(
            "Track wetness $w_l \\in [0.0, 1.0]$ represents surface water saturation (`0.0` = bone dry, `0.35` = damp/greasy, "
            "`0.70+` = standing water). Running dry slicks on a wet track—or wet tires on a dry track—incurs a severe "
            "hydroplaning or overheating penalty:"
        )
        col_w1, col_w2, col_w3 = st.columns(3)
        col_w1.info("**Dry Slicks ($w_l < 0.20$):** Optimal grip. Above $w_l = 0.20$, lap time degrades rapidly (+8s to +25s/lap).")
        col_w2.info("**Intermediate ($0.15 \\le w_l \\le 0.65$):** Optimal crossover tire for damp/light-rain conditions; overheats on dry tarmac.")
        col_w3.info("**Full Wet ($w_l > 0.60$):** Displaces standing water in heavy rain; +4s to +7s slower than Inters in light dampness.")

        st.subheader("E. Live Circuit Physics Constants (Currently Selected Round)")
        comp_rows = []
        for c_key in COMPOUNDS:
            comp = COMPOUNDS[c_key]
            override = circuit.compound_overrides.get(c_key, {})
            slope = override.get("deg_per_lap_sec", comp.deg_per_lap_sec * circuit.tire_wear_multiplier)
            cliff = override.get("cliff_lap", comp.cliff_lap)
            comp_rows.append(
                {
                    "Compound": f"{comp.name} ({comp.code})",
                    "Fresh Pace Offset vs. Soft": f"+{comp.base_offset_sec:.2f}s",
                    "Calibrated Wear Slope": f"{slope:.4f} s/lap",
                    "Thermal Cliff Onset": f"Lap {cliff} of stint",
                    "Wetness Operating Window": "Dry (w < 0.20)" if c_key in DRY_COMPOUNDS else ("Damp (0.15–0.65)" if c_key == "INTER" else "Heavy Rain (w > 0.60)"),
                }
            )
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

    # --- TAB 2: FIA SPORTING RULES ---
    with tab_rules:
        st.header("2. FIA Formula 1 Sporting Regulations Modeled")
        st.markdown(
            "The optimizer and strategy validator strictly enforce official FIA Formula 1 Sporting Regulations "
            "(Article 30.5 — Use of Tyres):"
        )

        r1, r2 = st.columns(2)
        with r1:
            st.markdown(
                """
                <div class="manual-card">
                <h4>📜 1. The Mandatory Two-Compound Rule (Dry Races)</h4>
                <p>Unless a driver uses <b>Intermediate</b> or <b>Full Wet</b> tires during the race, they <b>must use at least two different specifications of dry-weather tires</b> (e.g., Soft → Medium, Medium → Hard, or Soft → Hard).</p>
                <ul>
                  <li><b>0-Stop Dry Strategy:</b> Illegal under FIA rules (disqualification / +30s terminal penalty in DP).</li>
                  <li><b>Same-Compound 1-Stop (e.g., M → M):</b> Illegal unless rain tires are fitted later in the race.</li>
                  <li><b>Weather Exemption:</b> The moment a passing rain shower triggers an Intermediate or Wet stint, the 2-dry-compound constraint is automatically waived.</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with r2:
            st.markdown(
                """
                <div class="manual-card">
                <h4>⏱️ 2. Pit-Lane Speed Limit & Safety Car Discount</h4>
                <p>Entering the pit lane (60–80 km/h speed limit) costs a fixed circuit-calibrated <b>Green-Flag Pit Loss</b> (typically <b>19.0s – 25.0s</b> depending on pit-lane length and entry/exit geometry).</p>
                <ul>
                  <li><b>Green-Flag Stop:</b> Full pit delta lost while rivals circulate at full racing speed (~300+ km/h).</li>
                  <li><b>Safety Car / VSC Stop:</b> Field pace is neutralized (+40% slower lap times), reducing the relative time lost in the pit lane to roughly <b>55% of Green-Flag Pit Loss</b>—saving 8s–12s and creating prime strategic pit windows.</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="manual-card">
            <h4>⚡ 3. The Undercut vs. Overcut Strategic Trade-Off</h4>
            <ul>
              <li><b>Undercut:</b> Boxing 1–2 laps <i>before</i> a rival to exploit fresh-tire peak grip in clean air, erasing a 1.5s–2.5s gap before the rival responds.</li>
              <li><b>Overcut:</b> Staying out longer on durable tires when traffic behind a pit-stop exit would trap a pitting car, or when cold track temperatures delay tire warm-up on out-laps.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- TAB 3: BELLMAN DYNAMIC PROGRAMMING ---
    with tab_dp:
        st.header("3. Exact Backward Dynamic Programming (`engine/optimizer.py`)")
        st.markdown(
            "Rather than relying on greedy heuristics, `solve_optimal_strategy_dp()` computes the **provably globally "
            "optimal pit-stop sequence** using **Bellman Backward Dynamic Programming** across an exact multi-dimensional "
            "state space."
        )

        st.subheader("A. Markov State Space Definition")
        st.markdown("At any race lap $l \\in \\{1, \\dots, N\\}$, the car's strategic state $S$ is completely described by:")
        st.latex(r"S = (l,\; c,\; a,\; u,\; k)")
        st.markdown(
            "- **$l$**: Current race lap ($1 \\le l \\le N$).\n"
            "- **$c$**: Currently fitted tire compound ($\\text{SOFT}, \\text{MEDIUM}, \\text{HARD}, \\text{INTER}, \\text{WET}$).\n"
            "- **$a$**: Current tire stint age in laps ($0 \\le a \\le N$).\n"
            "- **$u \\in \\{0, 1\\}$**: Binary flag tracking whether at least two distinct dry compounds have been used (or wet tires fitted).\n"
            "- **$k \\in \\{0, \\dots, K_{\\max}\\}$**: Number of pit stops already executed."
        )

        st.subheader("B. Bellman Optimality Recursion")
        st.markdown(
            "Working backward from the checkered flag ($l = N + 1$) to the start ($l = 1$), the minimum remaining race "
            "time $V(l, c, a, u, k)$ satisfies:"
        )
        st.latex(
            r"V(l, c, a, u, k) = \min \begin{cases} "
            r"\underbrace{T_{\text{lap}}(l, c, a) + V(l+1, c, a+1, u, k)}_{\text{Action 1: Stay Out on Compound } c} \\[12pt] "
            r"\underbrace{\min_{c' \in \mathcal{C}} \Big[ T_{\text{lap}}(l, c', 0) + \Delta T_{\text{pit}}(l) + V(l+1, c', 1, u', k+1) \Big]}_{\text{Action 2: Pit at End of Lap } l \text{ for Compound } c' \;(k < K_{\max})} "
            r"\end{cases}"
        )
        st.markdown("With boundary condition at the checkered flag ($l = N + 1$):")
        st.latex(
            r"V(N+1, c, a, u, k) = \begin{cases} 0 & \text{if } u = 1 \text{ (FIA Legal)} \\ +\infty & \text{if } u = 0 \text{ (Illegal 1-Compound Dry Race)} \end{cases}"
        )

        st.subheader("C. Exact Stop-Count Families & Commutative Deduplication")
        st.markdown(
            "- **Exact Stop-Count Partitioning:** The solver tracks optimal policies independently for 1-stop, 2-stop, and 3-stop "
            "families so the Strategy Leaderboard always surfaces the best strategy within each pit-stop count.\n"
            "- **Commutative Stint Deduplication:** Because `Soft (18L) → Medium (39L)` and `Medium (39L) → Soft (18L)` produce "
            "identical total tire wear and fuel-independent stint durations (aside from slight fuel-weight interaction), the "
            "leaderboard deduplicates mirror permutations that sit within **0.15s** of each other to present genuinely distinct tactical options."
        )

    # --- TAB 4: MONTE CARLO & CHAOS ---
    with tab_mc:
        st.header("4. Monte Carlo Stochastic Rollout & Chaos Simulator")
        st.markdown(
            "A strategy that wins by 1.2s in a deterministic dry race can lose by 15s if a mid-race Safety Car gives a rival "
            "a cheap pit stop. The **Monte Carlo engine** (`run_monte_carlo_evaluation()`) evaluates robustness across "
            "$M \\in [100, 600]$ independent stochastic race universes:"
        )

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(
                """
                <div class="manual-card">
                <h4>🎲 Lap-by-Lap Pace Noise</h4>
                <p>Every lap samples zero-mean Gaussian execution noise $\\epsilon_l \\sim \\mathcal{N}(0, \\sigma_{\\text{lap}}^2)$ ($\\sigma \\approx 0.18\\text{s}$) for both player and rival, modeling micro-lockups, traffic, and wind gusts.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                """
                <div class="manual-card">
                <h4>🟡 Stochastic Safety Cars</h4>
                <p>Each simulation draws random Safety Car deployments (3–5 laps duration) calibrated to circuit incident probability, neutralizing field gaps and triggering opportunistic pit stops.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m3:
            st.markdown(
                """
                <div class="manual-card">
                <h4>🌧️ Passing Rain Cells</h4>
                <p>When enabled, random localized rain showers sweep across the circuit with a smooth half-sine wetness curve $w(t) = A \\sin(\\pi \\tau)$, testing intermediate crossover timing.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.subheader("Interpreting the P10 – P90 Confidence Fan Chart")
        st.markdown(
            "- **P90 (Top Edge of Shaded Band):** Best-case 90th-percentile outcome for the player (e.g., favorable Safety Car timing or rival traffic).\n"
            "- **P50 (Dashed Cyan Line):** Median gap across all Monte Carlo rollouts.\n"
            "- **P10 (Bottom Edge of Shaded Band):** Worst-case 10th-percentile downside risk.\n"
            "- **Win Probability (%):** Percentage of simulated races where the player crosses the finish line ahead of the rival ($\\text{Final Gap} > 0.0\\text{s}$)."
        )

    # --- TAB 5: TELEMETRY ETL PIPELINE ---
    with tab_etl:
        st.header("5. OpenF1 + FastF1 Telemetry Calibration Pipeline (`data/calibrate_fastf1.py`)")
        st.markdown(
            "All 23 rounds of the 2026 FIA Formula 1 World Championship are calibrated from real Formula 1 timing feeds "
            "via a dual-source ETL pipeline:"
        )
        st.markdown(
            """
            1. **Primary Feed — OpenF1 REST API (`api.openf1.org/v1`):**
               - Queries `/sessions`, `/session_result`, `/drivers`, `/stints`, and `/laps` with automatic HTTP 429 exponential backoff and `circuit_short_name` / `country_name` fallback matching.
               - Extracts official FIA race classifications (`P1–P20`, grid positions, laps completed, DNF statuses) and exact executed pit-stop laps and tire compound sequences into `data/actual_strategies.json`.
            2. **Secondary / Fallback Feed — FastF1 Timing Cache:**
               - Filters out Lap 1 standing-start laps, pit in/out laps, Safety Car / VSC laps, and outliers (`> 107%` median pace).
               - Fits ordinary least-squares (OLS) fuel-corrected linear degradation slopes per compound (`SOFT`, `MEDIUM`, `HARD`) for every circuit into `data/circuits.json` and `data/compound_profiles.json`.
            3. **Historical / New Circuit Anchors:**
               - Newly added or returning circuits (such as **Madring** or **Sepang**) use physics-scaled track geometry and surface roughness baselines until live session telemetry is ingested.
            """
        )

    # --- TAB 6: GLOSSARY ---
    with tab_glossary:
        st.header("6. Complete Glossary of Dashboard Metrics & Controls")
        glossary_data = [
            ("2026 Grand Prix Circuit", "Selects any of the 23 rounds (R01–R23), loading its calibrated lap count, base lap time, pit-lane delta, tire wear multiplier, and real GP results."),
            ("DP Solver Max Allowed Pit Stops", "Upper bound K_max (1 to 3 stops) on pit stops considered by the Bellman Dynamic Programming solver."),
            ("Player Pace Delta vs. Rival (s/lap)", "Intrinsic car + driver pace advantage (negative) or deficit (positive) applied to every green-flag lap."),
            ("Monte Carlo Rollouts", "Number of independent stochastic races simulated (100–600) to compute Win Probability and the P10–P90 confidence fan."),
            ("Current Race Lap (Decision Point)", "The exact lap at which live chaos triggers (Safety Car, Rival Undercut, or Rain Shower) are injected."),
            ("Deploy Safety Car on Current Lap", "Neutralizes the race for 4 laps starting on the Decision Point lap; in Auto DP mode, triggers live mid-race Bellman re-optimization."),
            ("Rival Boxes Early (Undercut Threat)", "Forces the rival car to pit 1 lap before the Decision Point onto Hard tires to test undercut vulnerability."),
            ("Passing Rain Shower (+8 Laps)", "Injects an 8-lap localized rain cell peaking at 0.45 track wetness, testing dry-to-intermediate crossover sensitivity."),
            ("Active Pit-Wall Strategy", "The currently active tire compound sequence and pit-stop laps for the player car, plus its time delta vs. the DP global optimum."),
            ("Projected Race Time", "Total deterministic elapsed race duration (in minutes) and final finish-line gap (+s = player finishes ahead of rival)."),
            ("Monte Carlo Win Probability", "Percentage of stochastic simulations where the player finishes ahead of the selected rival strategy."),
            ("Circuit Pit-Stop Loss (Green / SC)", "Total time lost entering/exiting the pit lane under green-flag conditions vs. under a Safety Car (~55% of green loss)."),
            ("Wear Index", "Circuit-specific surface abrasiveness multiplier (1.00x = championship average; >1.15x = high-degradation tracks like Bahrain/Suzuka/Sepang)."),
        ]
        st.dataframe(
            pd.DataFrame(glossary_data, columns=["Dashboard Metric / Control", "Engineering & Strategic Definition"]),
            use_container_width=True,
            hide_index=True,
        )


circuits = get_all_circuits(_file_mtime("data/circuits.json"))
actual_strategies_by_circuit = get_all_actual_strategies(_file_mtime("data/actual_strategies.json"))
sorted_keys = sorted(circuits.keys(), key=lambda k: (circuits[k].round, circuits[k].name))

# --- SIDEBAR: PAGE NAVIGATION & PIT-WALL CONTROLS ---
st.sidebar.title("🏎️ Pit-Wall Command Center")
active_page = st.sidebar.radio(
    "Workspace View",
    options=[
        "🏎️ Live Pit-Wall Simulator",
        "📘 Physics, FIA Rules & Math Manual",
    ],
    help="Switch between the interactive strategy simulator and the full engineering, physics & Bellman DP documentation.",
)

st.sidebar.markdown("---")
selected_key = st.sidebar.selectbox(
    "2026 Grand Prix Circuit (All 23 Rounds)",
    options=sorted_keys,
    index=0,
    format_func=lambda k: f"R{circuits[k].round:02d} — {circuits[k].name} ({circuits[k].country})",
    help="Select any of the 23 Grand Prix rounds. Loads calibrated lap count, base pace, pit-lane delta, tire wear curves, and real GP classifications.",
)
circuit = circuits[selected_key]
circuit_actuals = actual_strategies_by_circuit.get(selected_key, {})
actual_drivers = circuit_actuals.get("drivers", [])

# Render Manual Page if selected
if active_page == "📘 Physics, FIA Rules & Math Manual":
    render_engineering_manual(circuit)
    st.stop()

# --- SIMULATOR SIDEBAR CONTROLS ---
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Solver & Monte Carlo Settings")
max_stops = st.sidebar.slider(
    "DP Solver Max Allowed Pit Stops",
    min_value=1,
    max_value=3,
    value=2,
    help="Maximum number of pit stops (1, 2, or 3) the Bellman Dynamic Programming solver is permitted to schedule.",
)
driver_pace_offset = st.sidebar.slider(
    "Player Pace Delta vs. Rival (s/lap)",
    min_value=-0.40,
    max_value=0.40,
    value=0.00,
    step=0.05,
    help="Intrinsic car/driver pace offset per green-flag lap. Negative values (e.g., -0.20s) mean your car is faster than the rival; positive values mean slower.",
)
n_sims = st.sidebar.slider(
    "Monte Carlo Rollouts",
    min_value=100,
    max_value=600,
    value=250,
    step=50,
    help="Number of randomized race simulations executed to compute Win Probability and the P10–P90 gap confidence fan.",
)
enable_sc_mc = st.sidebar.checkbox(
    "Stochastic Safety Cars in Monte Carlo",
    value=True,
    help="When enabled, random Safety Car periods occur across Monte Carlo simulations based on historical circuit incident rates.",
)
enable_rain_mc = st.sidebar.checkbox(
    "Stochastic Rain Cells in Monte Carlo",
    value=True,
    help="When enabled, random passing rain showers occur across Monte Carlo simulations, testing wet/intermediate crossover risk.",
)

st.sidebar.markdown("---")
st.sidebar.subheader("🚨 Live Race Chaos Simulator")
chaos_lap = st.sidebar.slider(
    "Current Race Lap (Decision Point)",
    min_value=1,
    max_value=max(2, circuit.total_laps - 5),
    value=min(18, max(2, circuit.total_laps // 3)),
    help="Sets the exact race lap where live race disruptions (Safety Car, Rival Undercut, or Passing Rain Shower) are injected.",
)
inject_sc = st.sidebar.toggle(
    "🟡 Deploy Safety Car on Current Lap",
    value=False,
    help="Triggers a 4-lap Safety Car starting on the Decision Point lap (reducing pit-stop loss by ~45%). In Auto DP mode, triggers live mid-race Bellman re-optimization.",
)
inject_undercut = st.sidebar.toggle(
    "⚡ Rival Boxes Early (Undercut Threat)",
    value=False,
    help="Overrides the rival to pit 1 lap before the Decision Point onto fresh Hard tires to test whether your strategy survives an undercut.",
)
inject_rain = st.sidebar.toggle(
    "🌧️ Passing Rain Shower (+8 Laps)",
    value=False,
    help="Injects an 8-lap localized rain cell peaking at 0.45 surface wetness, penalizing dry slicks that stay out during the shower.",
)

# --- HEADER & TELEMETRY BADGE ---
st.title(f"🏁 R{circuit.round:02d}: {circuit.name}")
st.markdown(
    f'<div class="telemetry-badge">📡 Telemetry Source: {circuit.telemetry_source} '
    f'&nbsp;|&nbsp; {circuit.total_laps} Laps &nbsp;|&nbsp; Base Pace: {circuit.base_lap_time_sec:.2f}s '
    f'&nbsp;|&nbsp; Pit Delta: {circuit.pit_loss_sec:.2f}s</div>',
    unsafe_allow_html=True,
)

# --- SOLVE PRE-RACE DP & RIVAL BENCHMARKS ---
dp_result = solve_optimal_strategy_dp(
    circuit=circuit,
    max_stops=max_stops,
    driver_pace_offset_sec=driver_pace_offset,
)
rival_strategies = generate_standard_rival_strategies(circuit)

mode_options = [
    "🤖 Bellman DP Optimal (Auto)",
    "🛠️ Custom Pit-Wall Strategy (Manual)",
    "🏁 Real Grand Prix Result Strategy",
]

ctrl_col1, ctrl_col2 = st.columns([3, 2])
with ctrl_col1:
    strategy_mode = st.radio(
        "Player Pit-Wall Strategy Mode",
        options=mode_options,
        horizontal=True,
        help="Auto DP finds the mathematical optimum; Custom lets you build and stress-test your own pit stops; Real GP loads actual FIA race strategies.",
    )
with ctrl_col2:
    rival_idx = st.selectbox(
        "Primary Rival Strategy Benchmark",
        options=list(range(len(rival_strategies))),
        format_func=lambda i: rival_strategies[i].name,
        help="Select the baseline strategy your primary on-track rival executes in head-to-head race traces and Monte Carlo rollouts.",
    )
rival_plan = rival_strategies[rival_idx]

# --- CUSTOM OR REAL GRAND PRIX STRATEGY PANEL ---
custom_result = None
selected_actual_driver = None
base_player_plan = dp_result.plan

if strategy_mode == "🛠️ Custom Pit-Wall Strategy (Manual)":
    with st.expander("🛠️ Custom Pit-Wall Strategy Designer", expanded=True):
        b_col1, b_col2 = st.columns([1, 1])
        with b_col1:
            custom_start_comp = st.selectbox(
                "Starting Tire Compound",
                options=list(DRY_COMPOUNDS),
                index=0,
                format_func=lambda c: f"{COMPOUNDS[c].name} ({COMPOUNDS[c].code})",
                key=f"custom_start_{selected_key}",
                help="Tire compound fitted on the starting grid for Lap 1.",
            )
        with b_col2:
            custom_num_stops = st.number_input(
                "Scheduled Pit Stops",
                min_value=0,
                max_value=3,
                value=1,
                step=1,
                key=f"custom_stops_{selected_key}",
                help="Total number of scheduled pit stops (0 to 3). Note: FIA rules require at least 2 distinct dry compounds in dry races.",
            )

        custom_stops_input = []
        if custom_num_stops > 0:
            stop_cols = st.columns(int(custom_num_stops))
            for s_idx in range(int(custom_num_stops)):
                default_lap = max(
                    2,
                    min(
                        circuit.total_laps - 1,
                        int(round(circuit.total_laps * (s_idx + 1) / (custom_num_stops + 1))),
                    ),
                )
                default_comp_idx = 2 if (s_idx == custom_num_stops - 1 and custom_start_comp != "HARD") else 1
                with stop_cols[s_idx]:
                    st.markdown(f"**Pit Stop #{s_idx + 1}**")
                    p_lap = st.number_input(
                        f"Box on Lap (Stop {s_idx + 1})",
                        min_value=1,
                        max_value=circuit.total_laps - 1,
                        value=default_lap,
                        step=1,
                        key=f"custom_lap_{selected_key}_{s_idx}",
                        help=f"Race lap on which the car enters the pit lane for Pit Stop #{s_idx + 1}.",
                    )
                    p_comp = st.selectbox(
                        f"Fit Compound (Stop {s_idx + 1})",
                        options=list(DRY_COMPOUNDS),
                        index=default_comp_idx,
                        format_func=lambda c: f"{COMPOUNDS[c].name} ({COMPOUNDS[c].code})",
                        key=f"custom_comp_{selected_key}_{s_idx}",
                        help=f"Fresh tire compound fitted during Pit Stop #{s_idx + 1}.",
                    )
                    custom_stops_input.append((int(p_lap), p_comp))

        custom_result = build_custom_strategy_plan(
            circuit=circuit,
            start_compound=custom_start_comp,
            stops=custom_stops_input,
            driver_pace_offset_sec=driver_pace_offset,
        )
        base_player_plan = custom_result.plan

        if custom_result.is_fia_legal_dry:
            st.success(f"✅ FIA 2-Compound Dry Rule Satisfied — Active Plan: **{custom_result.stint_summary}**")
        else:
            for w in custom_result.warnings:
                st.warning(f"⚠️ {w}")

elif strategy_mode == "🏁 Real Grand Prix Result Strategy":
    with st.expander("🏁 Real Grand Prix Classification & Pit-Wall Stints (FastF1)", expanded=True):
        if actual_drivers:
            drv_idx = st.selectbox(
                f"Select Driver Classification ({circuit_actuals.get('source', 'FastF1 Race')})",
                options=list(range(len(actual_drivers))),
                format_func=lambda i: actual_drivers[i].display_label,
                key=f"actual_drv_{selected_key}",
                help="Load any classified driver's real Grand Prix tire compound sequence and exact pit-stop laps to benchmark against the Bellman DP optimum.",
            )
            selected_actual_driver = actual_drivers[drv_idx]
            custom_result = build_actual_driver_plan(
                circuit=circuit,
                driver_strategy=selected_actual_driver,
                driver_pace_offset_sec=driver_pace_offset,
            )
            base_player_plan = custom_result.plan

            m1, m2, m3, m4 = st.columns(4)
            pos_str = f"P{selected_actual_driver.position}" if selected_actual_driver.position < 90 else "DNF"
            grid_str = f"P{selected_actual_driver.grid_position}" if selected_actual_driver.grid_position > 0 else "Pit/NA"
            m1.metric(
                "Driver & Team",
                f"{selected_actual_driver.driver_name}",
                f"{selected_actual_driver.team}",
                help="Official driver and constructor entry from FIA / OpenF1 / FastF1 race classification.",
            )
            m2.metric(
                "Finish / Grid",
                f"{pos_str} (Started {grid_str})",
                selected_actual_driver.status,
                help="Official checkered-flag classification vs. starting grid position.",
            )
            m3.metric(
                "Laps Completed",
                f"{selected_actual_driver.laps_completed} / {circuit.total_laps}",
                help="Number of race laps completed by this driver relative to the scheduled Grand Prix distance.",
            )
            m4.metric(
                "Executed Pit Stops",
                f"{selected_actual_driver.num_stops} Stop(s)",
                selected_actual_driver.strategy_summary,
                help="Actual tire compound sequence and pit-stop laps executed during the Grand Prix.",
            )

            for w in custom_result.warnings:
                st.info(f"ℹ️ {w}")
        else:
            st.warning(
                "No real race classification is cached for this round yet. "
                "Run `python data/calibrate_fastf1.py` after the Grand Prix completes."
            )

# --- APPLY LIVE CHAOS SCENARIO ---
sc_laps = list(range(chaos_lap, min(circuit.total_laps + 1, chaos_lap + 4))) if inject_sc else []
wetness_profile = [0.0] * circuit.total_laps
if inject_rain:
    for l in range(chaos_lap - 1, min(circuit.total_laps, chaos_lap + 8)):
        progress = (l - (chaos_lap - 1)) / 7.0
        wetness_profile[l] = 0.45 * float(np.sin(np.pi * progress))

active_rival_plan = rival_plan
if inject_undercut:
    early_lap = max(3, chaos_lap - 1)
    active_rival_plan = CarStrategyPlan(
        name=f"Rival Early Undercut (M→H) [L{early_lap}]",
        start_compound="MEDIUM",
        pit_stops={early_lap: "HARD"},
    )

# Mid-race dynamic re-optimization if Safety Car is deployed at chaos_lap (Auto DP mode)
active_player_plan = base_player_plan
if inject_sc and strategy_mode == "🤖 Bellman DP Optimal (Auto)":
    curr_comp = base_player_plan.start_compound
    curr_age = 0
    used_two = False
    kept_stops = {}
    for l in range(1, chaos_lap):
        if l in base_player_plan.pit_stops:
            nxt = base_player_plan.pit_stops[l]
            if nxt != curr_comp:
                used_two = True
            curr_comp = nxt
            curr_age = 0
            kept_stops[l] = nxt
        else:
            curr_age += 1

    reopt = solve_optimal_strategy_dp(
        circuit=circuit,
        start_lap=chaos_lap,
        current_compound=curr_comp,
        current_tire_age=curr_age,
        used_different_compound=used_two,
        is_safety_car_now=True,
        max_stops=max_stops,
        driver_pace_offset_sec=driver_pace_offset,
    )
    merged_stops = {**kept_stops, **reopt.plan.pit_stops}
    codes = [COMPOUNDS[base_player_plan.start_compound].code] + [
        COMPOUNDS[merged_stops[l]].code for l in sorted(merged_stops)
    ]
    laps_label = ", ".join(f"L{l}" for l in sorted(merged_stops))
    active_player_plan = CarStrategyPlan(
        name=f"Live Re-Opt SC: {len(merged_stops)}-Stop ({'→'.join(codes)}) [{laps_label}]",
        start_compound=base_player_plan.start_compound,
        pit_stops=merged_stops,
        driver_pace_offset_sec=driver_pace_offset,
    )

# Deterministic head-to-head race trace & Monte Carlo rollout
det_trace = simulate_race_trace(
    circuit=circuit,
    player_plan=active_player_plan,
    rival_plan=active_rival_plan,
    safety_car_laps=sc_laps,
    wetness_by_lap=wetness_profile,
)

# Clean dry baseline time for custom/actual plan so we can benchmark directly vs. DP Optimal
clean_player_trace = simulate_race_trace(
    circuit=circuit,
    player_plan=base_player_plan,
    rival_plan=rival_plan,
)
clean_player_time_sec = float(clean_player_trace["player_cum_time"][-1])
delta_vs_dp_sec = clean_player_time_sec - dp_result.expected_total_time_sec

mc_summary = run_monte_carlo_evaluation(
    circuit=circuit,
    player_plan=active_player_plan,
    rival_plan=active_rival_plan,
    n_sims=n_sims,
    enable_safety_cars=enable_sc_mc,
    enable_weather=enable_rain_mc,
)

# --- KPI METRIC STRIP (WITH '?' TOOLTIPS ON EVERY METRIC) ---
k1, k2, k3, k4 = st.columns(4)
clean_plan_label = (
    active_player_plan.name
    .replace("DP Optimal: ", "")
    .replace("Live Re-Opt SC: ", "")
    .replace("Custom: ", "Custom | ")
)
is_benchmark_mode = strategy_mode in ("🛠️ Custom Pit-Wall Strategy (Manual)", "🏁 Real Grand Prix Result Strategy")
if is_benchmark_mode:
    k1_delta = f"{delta_vs_dp_sec:+.2f}s vs. DP Optimal"
else:
    k1_delta = "SC Re-Optimized" if inject_sc else "Bellman Global Optimum"

k1.metric(
    label="Active Pit-Wall Strategy",
    value=clean_plan_label,
    delta=k1_delta,
    delta_color="inverse" if (is_benchmark_mode and delta_vs_dp_sec > 0.05) else "normal",
    help="Active tire compound sequence and pit-stop laps (e.g., S→M→H [L16, L38]). Delta shows total dry-race time difference vs. the Bellman DP global optimum.",
)
k2.metric(
    label="Projected Race Time",
    value=f"{det_trace['player_cum_time'][-1] / 60.0:.2f} min",
    delta=f"{det_trace['final_gap_sec']:+.2f}s vs. Rival",
    help="Total simulated race duration under the active deterministic scenario. Positive delta (+s) means the player finishes ahead of the rival at the checkered flag.",
)
k3.metric(
    label="Monte Carlo Win Probability",
    value=f"{mc_summary.win_probability_pct:.1f}%",
    delta=f"Median Gap: {mc_summary.p50_gap_sec:+.2f}s",
    help="Percentage of stochastic Monte Carlo race simulations where the player beats the rival. Median Gap is the 50th-percentile (P50) finish-line gap.",
)
k4.metric(
    label="Circuit Pit-Stop Loss (Green / SC)",
    value=f"{circuit.pit_loss_sec:.1f}s / {circuit.pit_loss_sec * 0.55:.1f}s",
    delta=f"Wear Index: {circuit.tire_wear_multiplier:.2f}x",
    help="Time lost entering and exiting the pit lane under Green Flag vs. Safety Car (~55% of green loss). Wear Index scales circuit-specific tire degradation (1.00x = season average).",
)

st.markdown("---")

# --- ROW 1: MONTE CARLO FAN CHART & HEAD-TO-HEAD GAP ---
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader(
        "📈 Monte Carlo Gap Evolution (P10 – P90 Confidence Fan)",
        help="Tracks the lap-by-lap time gap (+s = player ahead) across hundreds of stochastic simulations. The shaded band spans the 10th to 90th percentiles; the solid red line is the deterministic scenario trace.",
    )
    laps_x = det_trace["laps"]
    p10_trace = np.percentile(mc_summary.all_gap_traces, 10, axis=0)
    p50_trace = np.percentile(mc_summary.all_gap_traces, 50, axis=0)
    p90_trace = np.percentile(mc_summary.all_gap_traces, 90, axis=0)

    fig_fan = go.Figure()
    fig_fan.add_trace(
        go.Scatter(
            x=np.concatenate([laps_x, laps_x[::-1]]),
            y=np.concatenate([p90_trace, p10_trace[::-1]]),
            fill="toself",
            fillcolor="rgba(0, 210, 190, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            name="P10–P90 Stochastic Band",
        )
    )
    fig_fan.add_trace(
        go.Scatter(
            x=laps_x,
            y=p50_trace,
            mode="lines",
            line=dict(color="#00D2BE", width=2.5, dash="dash"),
            name="Monte Carlo Median Gap",
        )
    )
    fig_fan.add_trace(
        go.Scatter(
            x=laps_x,
            y=det_trace["gap_to_rival"],
            mode="lines",
            line=dict(color="#FF3333", width=3),
            name="Deterministic Scenario Trace",
        )
    )
    fig_fan.add_hline(y=0.0, line_dash="dot", line_color="#8B949E")
    fig_fan.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        xaxis_title="Race Lap",
        yaxis_title="Gap to Rival (+s = Player Ahead)",
        height=390,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_fan, use_container_width=True)

with col_right:
    st.subheader(
        "🛞 Calibrated Tire Compound Degradation Curves",
        help="Shows lap-time loss (in seconds vs. a brand-new Soft tire) as stint age increases. Upward bends mark each compound's empirical thermal cliff.",
    )
    ages = np.arange(0, 41)
    fig_tire = go.Figure()
    for comp_key in DRY_COMPOUNDS:
        comp = COMPOUNDS[comp_key]
        deltas = [compute_tire_delta(comp_key, int(a), circuit, track_wetness=0.0) for a in ages]
        override = circuit.compound_overrides.get(comp_key, {})
        slope = override.get("deg_per_lap_sec", comp.deg_per_lap_sec * circuit.tire_wear_multiplier)
        fig_tire.add_trace(
            go.Scatter(
                x=ages,
                y=deltas,
                mode="lines",
                line=dict(color=comp.color, width=3),
                name=f"{comp.name} ({slope:.3f} s/lap)",
            )
        )
    fig_tire.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0E1117",
        plot_bgcolor="#161B22",
        xaxis_title="Tire Stint Age (Laps)",
        yaxis_title="Pace Delta vs. Fresh Soft (+s)",
        height=390,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_tire, use_container_width=True)

# --- ROW 2: DP STRATEGY LEADERBOARD (+ CUSTOM & REAL PODIUM BENCHMARKS) ---
st.subheader(
    "🏆 Strategy Leaderboard: DP Optimum vs. Real Grand Prix Podium",
    help="Ranks top Bellman Dynamic Programming strategies alongside your Custom Plan and actual Grand Prix podium finishers (P1–P3) under clean-air baseline conditions.",
)

all_entries = [(label, total_sec, plan, "DP Optimal") for label, total_sec, plan in dp_result.top_alternatives]

# Add Custom Strategy if in Custom mode
if strategy_mode == "🛠️ Custom Pit-Wall Strategy (Manual)" and custom_result is not None:
    legal_tag = "Custom Strategy" if custom_result.is_fia_legal_dry else "Custom (⚠️ FIA 1-Comp)"
    all_entries.append((custom_result.stint_summary, clean_player_time_sec, custom_result.plan, legal_tag))

# Automatically benchmark real Grand Prix Podium (P1, P2, P3) + currently selected real driver
included_driver_codes = set()
podium_candidates = [d for d in actual_drivers if d.position in (1, 2, 3) and d.laps_completed >= int(circuit.total_laps * 0.85)]
if selected_actual_driver is not None and selected_actual_driver.driver_code not in {d.driver_code for d in podium_candidates}:
    podium_candidates.append(selected_actual_driver)

for drv in podium_candidates:
    if drv.driver_code in included_driver_codes:
        continue
    included_driver_codes.add(drv.driver_code)
    drv_res = build_actual_driver_plan(
        circuit=circuit,
        driver_strategy=drv,
        driver_pace_offset_sec=driver_pace_offset,
    )
    drv_trace = simulate_race_trace(
        circuit=circuit,
        player_plan=drv_res.plan,
        rival_plan=rival_plan,
    )
    drv_time_sec = float(drv_trace["player_cum_time"][-1])
    pos_badge = f"P{drv.position}" if drv.position < 90 else "DNF"
    source_label = f"Real {pos_badge} ({drv.driver_code} • {drv.team})"
    all_entries.append((drv_res.stint_summary, drv_time_sec, drv_res.plan, source_label))

all_entries.sort(key=lambda item: item[1])
fastest_time_sec = all_entries[0][1]

leaderboard_rows = []
for rank, (label, total_sec, plan, source_tag) in enumerate(all_entries, start=1):
    delta_best = total_sec - fastest_time_sec
    leaderboard_rows.append(
        {
            "Rank": f"#{rank}",
            "Source": source_tag,
            "Starting Compound": COMPOUNDS[plan.start_compound].name,
            "Strategy Sequence": label,
            "Simulated Race Time (s)": f"{total_sec:.2f}s",
            "Delta vs. Fastest": "+0.00s" if abs(delta_best) < 1e-3 else f"+{delta_best:.2f}s",
        }
    )

st.dataframe(
    pd.DataFrame(leaderboard_rows),
    use_container_width=True,
    hide_index=True,
    column_config={
        "Rank": st.column_config.TextColumn("Rank", help="Overall speed ranking across DP optimal policies and real GP podium benchmarks."),
        "Source": st.column_config.TextColumn("Source", help="Whether the strategy comes from the Bellman DP solver, your Custom Builder, or an actual Grand Prix finisher."),
        "Starting Compound": st.column_config.TextColumn("Starting Compound", help="Tire compound fitted at race start (Lap 1)."),
        "Strategy Sequence": st.column_config.TextColumn("Strategy Sequence", help="Exact compound progression and scheduled pit-stop laps."),
        "Simulated Race Time (s)": st.column_config.TextColumn("Simulated Race Time (s)", help="Total simulated clean-air race duration in seconds."),
        "Delta vs. Fastest": st.column_config.TextColumn("Delta vs. Fastest", help="Time lost relative to the #1 fastest strategy on the board (+0.00s = global optimum)."),
    },
)