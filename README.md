Yamanote Line Train Dynamics Model

A first-principles physics simulation of the JR East Yamanote Line, compared
against the published timetable.

The simulation is not the point on its own. The point is the gap between what the
physics permits and what the railway actually schedules, and what that gap can be
made to measure.

---

## Result

Simulating all 30 interstation segments from published rolling stock
specifications gives a **minimum circuit time of 2,309 s**. Applying the two
layers of conservatism Japanese operators document — drawing the run curve below
the posted speed limit, then rounding the result up — gives a **standard run time
of 2,390 s**.

The published circuit time is **3,948 s**. The raw gap is **1,639 s, or 42%**.

That gap is not a recovery-margin measurement. It is dominated by station dwell,
which JR East does not publish. Circuit time = run time + margin + dwell is one
equation in two unknowns, so **margin and dwell are not separately identifiable
from public data**.

Running the inference in the direction the evidence supports — margin practice is
documented, dwell is not — and solving for dwell gives:

| Assumed margin | Implied mean dwell (brake 4.2 km/h/s) | (brake 3.0 km/h/s) |
|---|---|---|
| 3% | 49.5 s | 45.1 s |
| 5% | 48.0 s | 43.4 s |
| 8% | 45.6 s | 40.9 s |
| 10% | 44.0 s | 39.2 s |

**Implied mean station dwell is 39–50 s.**

The estimate is well-conditioned: physics error enters the circuit equation
multiplied by the run time (2,390 s), while dwell is multiplied by 30 stations, so
a 5% physics error moves implied dwell by about 4 s. Changing the braking
assumption by 29% moves it by roughly 4 s; sweeping the margin across its entire
plausible range moves it by about 5 s.

Peer operators surveyed by MLIT report base dwell of 15–30 s, rising to 40–60 s at
major stations. An estimate near 45 s is at the top of that range, consistent with
a line serving the two busiest stations in the world.

**Assumption-free bound.** A mean dwell of 51.9 s or more would imply no recovery
margin at all. MLIT records that JR East explicitly adds margin, so that is
excluded — an upper bound on dwell requiring no modelling assumption.

---

## Installing and running

```bash
python -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -e ".[app]"
```

Launch the application (Windows: double-click `run.bat`):

```bash
.venv\Scripts\python.exe launcher.py
```

The launcher picks a free port, starts the server, waits until it is serving, and
opens a native desktop window via `pywebview`. `--browser` opens a browser tab
instead; `--frameless` removes the window title bar.

Analysis scripts must be run with the virtual environment's interpreter:

```bash
.venv\Scripts\python.exe scripts\validate.py
```

| Script | Output |
|---|---|
| `smoke_check.py` | Physics sanity checks against published figures |
| `convergence.py` | Integrator step-size study |
| `validate.py` | Circuit decomposition and dwell inference |
| `fleet.py` | Cross-class comparison |
| `energy.py` | Energy accounting and balance check |
| `coasting.py` | Four-phase solver and Pareto front |
| `gradient.py` | Bounding study for the flat-track assumption |
| `measure_dwell.py` | Live dwell measurement (requires an ODPT key) |

Each script calls `scripts/_env.py:ensure()` first, which reports the correct
interpreter if the wrong one is used.

---

## Method

| Quantity | Expression |
|---|---|
| Equation of motion | `m_eff · dv/dt = F(v) − R_run(v) − R_grade − R_curve` |
| Effective mass | `m_eff = m(1 + λ)`, λ ≈ 0.08 |
| Running resistance | `R(v) = A + Bv + Cv²` |
| Aerodynamic term | `C = ½ρ(C_d,ends + C_d,car · n)A_f` |
| Grade resistance | `R = mg·sin θ ≈ mg·(‰/1000)` |
| Tractive effort | `F(v) = min(F_max, P/v, μ(v)·m_adh·g)` |
| Starting effort | `F_max = m_eff·a_start + R(0)` |
| Adhesion | `μ(v) = 0.161 + 7.5/(v_kmh + 44)` (Curtius–Kniffler) |
| Jerk-limited stop | `T = v₀/a_max + a_max/j` |
| Brake application | `x + d_stop(v) ≥ L` |
| Traction energy | `E = ∫F·v dt` |
| Energy balance | `E_traction = E_resistance + E_brake` |

Integration is fixed-step RK4 with explicit control-phase switching. Root finding
is bisection. The braking solution is closed-form. scipy is not a dependency.

The brake application point is found by intersecting the forward run curve with a
brake curve solved backwards from rest at the platform. A forward-only solver
overshoots the platform.

---

## Numerical validation

| Check | Result |
|---|---|
| Final position error | 0.0 m on all 30 segments |
| Circuit time convergence | 8 ms across dt = 0.5 → 0.005 s |
| Energy balance residual | 2.5 × 10⁻⁴, first order in dt |

The energy balance is a genuine test rather than a restatement: a segment starts
and ends at rest, so `E_traction = E_resistance + E_brake` must hold, and both
sides are computed independently. It initially failed at 7.3%, caused by cruise
force being recorded as `effort_n`, which returns zero at maximum speed — where
cruise occurs on this line. Run times were unaffected; every energy figure was
wrong until corrected.

---

## Energy

One circuit at nominal loading:

| Destination | kWh | Share of traction work |
|---|---|---|
| Traction work at the wheel | 1,221.2 | 100% |
| → running resistance | 94.9 | 7.8% |
| → dissipated in braking | 1,125.8 | 92.2% |
| Auxiliary load | 136.2 | 11.2% |
| Regenerated and reused | −667.6 | −54.7% |
| **Net from supply** | **689.9** | |

**92% of traction work is dissipated in braking**, a direct consequence of 30
stops in 34.5 km. Regeneration therefore recovers more here than on a
long-distance line, where traction work goes to aerodynamic drag.

Net specific consumption is **1.82 kWh per car-km**, against a published
heavy-metro band of 2–4. **That is below the band, not inside it.**

This exposes a conflict between two published anchors:

- At the assumed regeneration receptivity of 0.70, consumption is 1.82 — below the
  band. Receptivity of 0.3–0.5 moves it inside.
- Matching JR East's published 回生率 of 59.0% for this class on this line
  requires a *higher* receptivity, about 0.76. The model produces 54.7%.

The two anchors disagree about the same unpublished parameter. Neither is strong
enough to override the other: the energy band is a range across operators rather
than a measurement of this line, and the 回生率 denominator could not be verified
at source. The conflict is recorded, not resolved, and the default is left
untuned at 0.70.

---

## Optimal coasting

The energy-minimal profile for a fixed distance and time is accelerate, cruise,
coast, brake. With the shape fixed the strategy reduces to one parameter — the
coast onset — found by bisection on run time.

The comparison holds run time equal. Beating a flat-out run proves nothing, since
the flat-out run is faster. The alternative to coasting is cruising slower.
Shinagawa → Ōsaki, 2,000 m:

| Run time | Coast | Cruise slower | Coasting saves |
|---|---|---|---|
| 102% | 25.03 kWh | 25.75 kWh | +2.8% |
| 110% | 20.73 kWh | 21.58 kWh | +4.0% |
| 130% | 16.50 kWh | 17.25 kWh | +4.3% |

**The shape of the front matters more than the strategy along it.** Coasting beats
cruising slower by 3–4%. Allowing 10% more run time cuts circuit energy by about a
third, since kinetic energy scales with the square of speed.

Coasting and regeneration are partial substitutes: the saving falls from 5.3% at
zero receptivity to 2.5% at perfect receptivity.

JR East reports 12% and 15.7% energy savings from optimal driving technique on
this line. That is a broader intervention than choosing a coast onset — it also
covers avoiding unnecessary acceleration and reducing variance between drivers,
which a single-trajectory optimiser cannot represent.

---

## Fleet comparison

Five classes under an identical comfort jerk limit. Every performance figure is
published.

| Class | Formation | Mass | Power | v_max | Accel |
|---|---|---|---|---|---|
| E235 (Yamanote) | 6M5T, 11 cars | 340.8 t | 3,360 kW | 90 km/h | 3.00 km/h/s |
| E233 (Chūō Rapid) | 6M4T, 10 cars | 318.8 t | 3,360 kW | 100 km/h | 3.00 km/h/s |
| E5 (Tōhoku) | 8M2T, 10 cars | 453.5 t | 9,600 kW | 320 km/h | 1.71 km/h/s |
| E7/W7 (Hokuriku) | 10M2T, 12 cars | 540.0 t | 12,000 kW | 275 km/h | 1.60 km/h/s |
| N700S (Tōkaidō) | 14M2T, 16 cars | < 700 t | 17,080 kW | 285 km/h | 2.60 km/h/s |

Power per passenger separates the categories more sharply than power per tonne:
13 kW/passenger for a Shinkansen against 2 kW/passenger for a commuter EMU, where
power-to-weight differs by only a factor of two.

**Speed-dependent braking.** The E7 publishes a deceleration falling from
2.69 km/h/s below 70 km/h to 1.44 at 275 km/h, because regenerative brake effort
is power-limited as tractive effort is. Integrating the real curve gives a
**5,670 m** stop from 275 km/h; assuming the low-speed rate throughout gives
3,943 m, understating by 30%. A constant service brake rate is safe for a 90 km/h
commuter EMU because the whole speed range sits in the flat part of the curve. It
is not safe for high-speed stock.

N700S formation mass is published only as "under 700 t", so it is carried as a
bound and its power-to-weight is reported as a lower bound.

---

## Gradient

The headline results assume level track. Station elevations were sampled from a
terrain model and combined with published kilometrage to derive a per-segment
gradient. **That profile is deliberately not used as a model input**: railways are
graded so track does not follow terrain, so ground elevation systematically
overstates track gradient.

It is used as a bound instead.

| | Circuit run time | vs flat | Net energy |
|---|---|---|---|
| Flat (headline model) | 2,309.0 s | — | 649.9 kWh |
| Graded, outer loop | 2,319.7 s | +10.7 s | 658.5 kWh |
| Graded, inner loop | 2,316.9 s | +8.0 s | 668.7 kWh |

Run time does not cancel even though gravitational work does: climbing is
power-limited, so it costs more time than the matching descent returns, and on a
descent the brakes must also work against gravity.

Under gradients known to be exaggerated, circuit run time moves **0.46%** while the
worst single segment shifts **+7.7 s**. The flat-track assumption is therefore safe
for the circuit-level dwell inference and unsafe for any per-segment claim.

The line's published maximum gradient is 34‰ between Tabata and Nishi-Nippori.

---

## Data sources

| Quantity | Source |
|---|---|
| Station sequence, operating distance (営業キロ) | [Yamanote Line, Wikipedia](https://en.wikipedia.org/wiki/Yamanote_Line) |
| Maximum gradient, line speed | [山手線, Wikipedia](https://ja.wikipedia.org/wiki/山手線) |
| E235-0 specifications | [JR東日本E235系電車](https://ja.wikipedia.org/wiki/JR東日本E235系電車) |
| E233-0 specifications | [JR東日本E233系電車](https://ja.wikipedia.org/wiki/JR東日本E233系電車) |
| E5 specifications | [新幹線E5系電車](https://ja.wikipedia.org/wiki/新幹線E5系電車) |
| E7/W7 specifications | [新幹線E7系・W7系電車](https://ja.wikipedia.org/wiki/新幹線E7系・W7系電車) |
| N700S specifications | [新幹線N700S系電車](https://ja.wikipedia.org/wiki/新幹線N700S系電車) |
| Recovery margin and dwell practice, all operators | [MLIT 運行計画 survey (PDF)](https://www.mlit.go.jp/kisha/kisha05/08/080722_3/01.pdf) |
| Station boardings FY2024 | [JR East 各駅の乗車人員](https://www.jreast.co.jp/company/data/passenger/) |
| Peak congestion 125%, Ueno→Okachimachi | [MLIT 混雑率 survey FY2023](https://ueno.keizai.biz/headline/752/) |
| Regeneration rate 59.0% | [JR East press material](https://www.jreast.co.jp/press/2024/20240508_ho01.pdf) |
| Optimal-driving savings 12% / 15.7% | [JSME 鉄道分野における省エネ技術](https://www.jsme.or.jp/kaisi/1240-13/) |
| Station coordinates (display only) | [OpenStreetMap via Overpass](https://overpass-api.de/api/interpreter) |
| Station elevations (bounding study) | [Open-Elevation](https://api.open-elevation.com/api/v1/lookup) |
| Real-time train positions | [ODPT](https://api.odpt.org/api/v4/odpt:Train) |

Every parameter in `data/` is tagged `published`, `derived`, `modelled`,
`bounded` or `unknown`, and the tag is carried through the code.

### Access limitations

- JR East returns HTTP 403 to automated requests for its own ridership page and
  press PDFs. Those figures reach this project through secondary compilations,
  and the 回生率 denominator could not be confirmed at source.
- ODPT real-time data for JR East requires a registered consumer key. The
  unauthenticated mirror carries Toei only, so the Yamanote cannot be measured.
- Per-segment scheduled times are published only to the minute. A ±30 s
  quantization exceeds the effect being measured, so no per-segment timetable is
  committed.

---

## Assumptions and limitations

**Assumptions**

- Level track. Known false on at least one segment; bounded above.
- No signal or ATC restrictions. Not public, and the largest unmodelled
  contributor to run time.
- No per-segment curve or turnout speed limits.
- Dry rail, no wind.
- Full tractive effort throughout acceleration.
- Published deceleration treated as a net rate.
- One-hour power rating rather than short-term overload capability. This makes
  the model slower than the real machine in the mid-speed region, so any measured
  fast bias is a lower bound.

**Limitations**

- Dwell is inferred, not measured. If true run time is materially higher than
  modelled — most plausibly through unmodelled ATC — implied dwell falls
  proportionally. A 17% error moves it from ~45 s to ~34 s.
- Davis coefficients are modelled. No published set exists for this class; they
  are estimated from mass, car count and frontal area, and are the largest single
  uncertainty in the physics core.
- Per-segment claims are unsafe: minute-rounded schedules and unmodelled gradient
  both bite at that scale.
- Shinkansen aerodynamics do not follow from the commuter Davis estimator, which
  is why those classes are refused rather than simulated on this line.
- Regeneration receptivity is unpublished and moves net energy by a factor of
  three.

---

## Repository layout

```
app.py                     Streamlit application
launcher.py                Starts the server, opens a native window
run.bat                    Windows entry point
data/
  yamanote_stations.csv    30 stations, published kilometrage
  station_coordinates.csv  OpenStreetMap coordinates, display only
  gradients.csv            Derived terrain profile, bounding study only
  rolling_stock.json       E235-0, provenance-tagged
  fleet.json               Comparison classes, bounds and gaps preserved
  timetable_anchors.json   Circuit times, minute-resolution limitation
  operational_practice.json  MLIT survey: margin and dwell practice
  ridership.csv            JR East boardings per station, FY2024
  congestion.json          MLIT peak congestion, critical sections only
  regeneration.json        JR East regeneration figures
src/tokyoline/
  units.py                 km/h/s to m/s2 conversion
  stock.py                 TrainSpec, effective mass, adhesion
  resistance.py            Davis, grade, curve
  traction.py              Effort curve, capability vs applied force
  brake.py                 Jerk-limited braking, closed form
  segment.py               RK4 integrator, control-phase switching
  network.py               Loop geometry, whole-circuit simulation
  energy.py                Traction energy, losses, regeneration
  coasting.py              Four-phase solver, Pareto front
  fleet.py                 Cross-class comparison, bounded quantities
  conditions.py            Rail surface condition and braking adhesion
  validate.py              Decomposition ladder, identifiability locus
  odpt.py                  Real-time client, censoring-aware dwell
scripts/                   Analysis entry points
```

---

## Application

Two views. **Line** carries the departure board, the model-versus-published
comparison, the 2D map and an animated 3D track view with synchronised instrument
panels for speed, acceleration, tractive effort, power and running resistance.
**Technical** carries the segment profiles, traction curves, energy breakdown,
coasting Pareto front, validation ladder, parameter provenance and sources.

The departure board is generated from the physics model plus an assumed headway
and dwell. It is a modelled schedule, not a live feed, and states so.

Time of day is the primary control, setting headway, crowding and dwell together.
Only the 125% peak congestion figure is published; headway and dwell are modelled.

Note that Streamlit hot-reloads `app.py` but does not re-import the `tokyoline`
package. After editing anything under `src/`, restart the server rather than
refreshing the browser.

---

## Status

Complete: data layer, physics core, timetable validation, fleet comparison,
energy accounting, optimal coasting, and the application.

Outstanding:

1. Measured dwell. The ODPT pipeline is built and validated against another
   operator's live feed; it requires a registered consumer key.
2. Signal and ATC restrictions, not public at usable precision.
3. Verification of the 回生率 denominator, which would let receptivity be set
   from a measurement.
4. Speed-dependent braking in the segment solver, which would allow the
   high-speed classes to be simulated.
  y a m a n o t e - d y n a m i c s 
 
 
