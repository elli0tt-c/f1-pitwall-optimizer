# 🏎️ F1 / Endurance Pit-Wall Strategy Simulator (`v2.0 — 2026 FastF1 Edition`)

An interactive motorsport race-strategy optimizer and stochastic pit-wall decision engine built in Python. Combines **real Formula 1 telemetry calibration (`fastf1`)** across all **23 circuits of the 2026 FIA Formula 1 World Championship** with **exact Backward Dynamic Programming (Bellman optimality)** and **Monte Carlo stochastic race simulations**.

---

## ✨ Key Capabilities

1. **Automated FastF1 Telemetry ETL & Calibration Pipeline (`data/calibrate_fastf1.py`)**
   - Extracts real Grand Prix race sessions (`R`) across all **23 rounds of the 2026 F1 calendar** using crash-isolated worker processes and local disk caching (`data/fastf1_cache/`).
   - Filters out Lap 1, pit-in/pit-out laps, Safety Car / VSC periods (`TrackStatus != '1'`), and 107% outliers.
   - Removes fuel-mass burn effects to isolate pure **clean-air base lap pace** (10th percentile), **empirical pit-lane time loss** (median pit-in + pit-out delta vs. green-flag median), **Safety Car deployment probability**, and **per-compound linear degradation slopes** (`SOFT`, `MEDIUM`, `HARD`).

2. **Lap-by-Lap Vehicle & Tire Physics Engine (`engine/physics.py`)**
   - **Fuel Mass Burn:** Linear fuel burn model reducing car mass and lap time over the race distance.
   - **Compound Degradation & Thermal Cliff:** Circuit-specific empirical pace offsets and linear rubber wear combined with a non-linear quadratic performance cliff beyond each compound's thermal life:
     $$\Delta t_{\text{tire}}(c, a) = \Delta t_{0,c} + \alpha_c \cdot a + \beta_c \cdot \max\left(0, a - a_{\text{cliff},c}\right)^2 + \Delta t_{\text{wet}}(c, w)$$
   - **Wake Aerodynamics & Overtaking Thresholds:** Models dirty-air downforce loss when trailing within `1.0s` unless net pace advantage overcomes circuit-specific overtaking difficulty.
   - **Safety Car Neutralization:** Compresses field gaps and discounts pit-stop delta by ~45% during Safety Car windows.

3. **Exact Backward Dynamic Programming Solver (`engine/optimizer.py`)**
   - Solves the finite-horizon Bellman optimality equations across state space `(lap, compound, tire_age, used_two_compounds, stops_used)` to find the globally optimal pit-stop lap(s) and compound sequence.
   - Strictly enforces the sporting regulation requiring at least **two distinct dry compounds** in a dry race.
   - Supports **live mid-race re-optimization** from any race lap and tire state (e.g., instantaneous pit-wall reaction to a Safety Car deployment).

4. **Monte Carlo Stochastic Risk Engine & Interactive Pit-Wall UI (`app.py`)**
   - Stress-tests candidate strategies across hundreds of stochastic race futures featuring random Safety Car windows, passing rain cells (with automatic Dry $\leftrightarrow$ Intermediate $\leftrightarrow$ Full Wet crossover rules), and Gaussian lap-time variance.
   - Dark-mode **Streamlit + Plotly** command center with live chaos toggles, P10–P90 confidence fan charts, calibrated compound degradation curves, and a full DP strategy leaderboard.

---

## 📁 Project Architecture

```text
f1-pitwall-optimizer/
├── data/
│   ├── calibrate_fastf1.py     # FastF1 ETL & empirical parameter calibration script (23 circuits)
│   ├── circuits.json           # Calibrated 2026 circuit boundary conditions (R01–R23)
│   └── compound_profiles.json  # Circuit-specific empirical Soft/Medium/Hard wear fits
├── engine/
│   ├── __init__.py
│   ├── physics.py              # Fuel, tire cliff, wet crossover, dirty air, and race trace physics
│   └── optimizer.py            # Backward DP solver, mid-race re-solver, and Monte Carlo engine
├── app.py                      # Streamlit + Plotly interactive pit-wall dashboard
├── requirements.txt            # numpy, scipy, pandas, plotly, streamlit, fastf1
├── LICENSE                     # MIT License
└── README.md

🚀 Quickstart (Local Setup)

1. Clone & Create Virtual Environment
git clone [https://github.com/](https://github.com/)<your-username>/f1-pitwall-optimizer.git
cd f1-pitwall-optimizer
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

2. Run FastF1 Telemetry Calibration (Optional — Pre-Calibrated JSONs Included)
python data/calibrate_fastf1.py

3. Launch the Interactive Pit-Wall Dashboard
streamlit run app.py

🛠️ Tech Stack
Telemetry & Data Engineering: fastf1, pandas, numpy
Optimization & Simulation: Exact Backward Dynamic Programming (NumPy), Monte Carlo Stochastic Rollouts
Interactive Visualization: streamlit, plotly

📜 License
Released under the MIT License.