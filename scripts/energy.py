"""Phase 5: energy accounting over the Yamanote circuit."""

from __future__ import annotations

from _env import ensure

ensure()

from tokyoline import build_model, load_segments, simulate_segment  # noqa: E402
from tokyoline.energy import (  # noqa: E402
    AUXILIARY_KW,
    JOULES_PER_KWH,
    REGEN_CUTOFF_KMH,
    REGEN_EFFICIENCY,
    REGEN_RECEPTIVITY,
    circuit_energy,
    segment_energy,
)
from tokyoline.validate import load_practice  # noqa: E402

traction, brake = build_model(load_factor=1.0)
segments = load_segments()
dwell = load_practice()["yamanote_dwell_scenarios_seconds"]["central"]

results = [
    simulate_segment(traction, brake, s.distance_m,
                     speed_limit_ms=traction.spec.max_speed_ms, dt=0.05)
    for s in segments
]


def kwh(j: float) -> float:
    return j / JOULES_PER_KWH


print("=" * 78)
print("PHASE 5  --  ENERGY ACCOUNTING")
print("=" * 78)
print(f"  {len(segments)} segments, {sum(s.distance_m for s in segments)/1000:.1f} km, "
      f"dwell {dwell:.0f} s")
print(f"  effective mass {traction.spec.effective_mass_kg(traction.load_factor)/1000:.0f} t "
      f"(load factor {traction.load_factor})")

# ---------------------------------------------------------------- balance
print()
print("=" * 78)
print("ENERGY BALANCE CHECK")
print("=" * 78)
print("  Each segment starts and ends at rest, so the kinetic term drops out of")
print("  the work-energy theorem and this identity must hold exactly:")
print()
print("      E_traction  =  E_resistance  +  E_brake_system")
print()
print("  Both sides are computed independently, so agreement tests the integrator.")
print()
print(f"  {'segment':<40} {'traction':>10} {'residual':>11} {'rel err':>10}")
print("  " + "-" * 74)
worst = 0.0
for seg, r in zip(segments, results):
    b, _ = segment_energy(r, traction, auxiliary_kw=0.0)
    worst = max(worst, b.balance_error_fraction)
    label = f"{seg.from_station} → {seg.to_station}"
    print(f"  {label:<40} {kwh(b.traction_j):8.2f}kWh {b.balance_residual_j:9.2e}J "
          f"{b.balance_error_fraction:9.2e}")
print()
print(f"  worst relative balance error: {worst:.2e}")
print()
print("  It converges at first order with step size, which is what a trapezoidal")
print("  integral across the force discontinuities at the phase boundaries should")
print("  do -- evidence the residual is discretisation and not a modelling error:")
print()
for step in (0.2, 0.1, 0.05, 0.02, 0.01):
    w = 0.0
    for s in segments:
        rr = simulate_segment(traction, brake, s.distance_m,
                              speed_limit_ms=traction.spec.max_speed_ms, dt=step)
        bb, _ = segment_energy(rr, traction, auxiliary_kw=0.0)
        w = max(w, bb.balance_error_fraction)
    print(f"    dt = {step:5.3f} s  ->  {w:.3e}")

# ---------------------------------------------------------------- circuit
total, profile = circuit_energy(results, traction, dwell_s=dwell)

print()
print("=" * 78)
print("WHERE THE ENERGY GOES  (one full circuit)")
print("=" * 78)
print(f"  {'destination':<34} {'kWh':>10} {'share of traction':>19}")
print("  " + "-" * 66)
print(f"  {'Traction work at the wheel':<34} {kwh(total.traction_j):10.1f} "
      f"{'100.0%':>19}")
print(f"    {'-> running resistance':<32} {kwh(total.resistance_j):10.1f} "
      f"{100*total.resistance_j/total.traction_j:18.1f}%")
print(f"    {'-> absorbed by brakes':<32} {kwh(total.brake_system_j):10.1f} "
      f"{100*total.brake_system_j/total.traction_j:18.1f}%")
print()
print(f"  {'Auxiliary (hotel) load':<34} {kwh(total.auxiliary_j):10.1f} "
      f"{100*total.auxiliary_j/total.traction_j:18.1f}%")
print(f"  {'Regenerated and reused':<34} {-kwh(total.regenerated_j):10.1f} "
      f"{-100*total.regenerated_j/total.traction_j:18.1f}%")
print("  " + "-" * 66)
print(f"  {'NET drawn from the supply':<34} {kwh(total.net_j):10.1f}")
print()
n_cars = traction.spec.n_cars
pax = traction.load_factor * traction.spec.capacity
print(f"  gross (traction + auxiliary)   {kwh(total.gross_j):.1f} kWh")
print(f"  regeneration saving            {100*total.regen_saving_fraction:.1f}% of gross")
print(f"  net specific consumption       {total.kwh_per_km():.1f} kWh per route-km")
print(f"                                 {total.kwh_per_car_km(n_cars):.2f} kWh per car-km")
print(f"  at {pax:.0f} passengers aboard      "
      f"{total.wh_per_passenger_km(pax):.1f} Wh per passenger-km")
per_car = total.kwh_per_car_km(n_cars)
print()
print(f"  Published heavy-metro consumption is typically 2-4 kWh per car-km.")
if 2.0 <= per_car <= 4.0:
    print(f"  {per_car:.2f} sits inside that band.")
else:
    side = "BELOW" if per_car < 2.0 else "ABOVE"
    print(f"  {per_car:.2f} sits {side} that band, not inside it.")
    if per_car < 2.0:
        print(f"  It is {100*(1 - per_car/2.0):.0f}% under the bottom at the assumed")
        print(f"  receptivity of {REGEN_RECEPTIVITY}. At 0.3-0.5 it moves inside --")
        print("  which pulls AGAINST the 回生率 cross-check below, where matching")
        print("  the published 59% needs a HIGHER receptivity of about 0.76.")
        print("  The two published anchors disagree about the same parameter.")
        print("  This project does not resolve that; it records it.")

# ------------------------------------------------------------- resistance
print()
print("=" * 78)
print("WHY SO MUCH GOES TO THE BRAKES")
print("=" * 78)
frac_brake = total.brake_system_j / total.traction_j
print(f"  {100*frac_brake:.0f}% of traction work ends up in the brakes rather than")
print("  overcoming resistance. That is a direct consequence of the duty cycle:")
print("  30 stops in 34.5 km means the train is repeatedly accelerating a 480 t")
print("  mass to 90 km/h and then throwing that kinetic energy away.")
print()
print("  It is also why regeneration matters far more on a metro than on a")
print("  long-distance line, where most traction work goes to aerodynamic drag")
print("  and is unrecoverable by any means.")

# --------------------------------------------------------------- regen
print()
print("=" * 78)
print("REGENERATION SENSITIVITY  (all three factors are MODELLED)")
print("=" * 78)
print(f"  defaults: efficiency {REGEN_EFFICIENCY}, receptivity {REGEN_RECEPTIVITY}, "
      f"cutoff {REGEN_CUTOFF_KMH} km/h")
print()
print(f"  {'receptivity':>12} {'net kWh':>10} {'saving':>9} {'kWh/car-km':>12}")
print("  " + "-" * 48)
for rec in (0.3, 0.5, 0.7, 0.9, 1.0):
    t2, _ = circuit_energy(results, traction, dwell_s=dwell, regen_receptivity=rec)
    print(f"  {rec:12.2f} {kwh(t2.net_j):10.1f} "
          f"{100*t2.regen_saving_fraction:8.1f}% {t2.kwh_per_car_km(n_cars):12.2f}")
print()
print("  Receptivity is the least defensible number in the model. It is the")
print("  fraction of returned energy another train actually absorbs, and on a")
print("  line with 2.5 minute headway it should be high -- but it is not")
print("  published, and the net figure moves by a third across this range.")

print()
print("=" * 78)
print("EXTERNAL CROSS-CHECK  --  JR East's published regeneration rate")
print("=" * 78)
import json  # noqa: E402
from tokyoline.data import DATA_DIR  # noqa: E402

with open(DATA_DIR / "regeneration.json", encoding="utf-8") as fh:
    reg = json.load(fh)["regeneration_rate"]

published = reg["values_percent"]["all_day_average"]
modelled = 100 * total.regenerated_j / total.traction_j
print(f"  JR East publishes a 回生率 of {published:.1f}% for the E235 on this line")
print(f"  ({reg['predecessor_comparison']['value_percent']:.0f}% for the E231 it replaced).")
print()
print(f"  this model, regenerated / traction: {modelled:.1f}%")
print(f"  at the assumed receptivity of       {REGEN_RECEPTIVITY:.2f}")
print()
implied = REGEN_RECEPTIVITY * published / modelled
print(f"  matching {published:.1f}% exactly would need receptivity ~{implied:.2f}")
print()
print("  That is agreement to within about 8% on a parameter that was otherwise")
print("  a pure guess. But treat it carefully: 回生率 is a RATIO and its exact")
print("  denominator could not be confirmed at source, because JR East blocks")
print("  automated retrieval of its own documents. The figure is used here as a")
print("  cross-check only. The model default is NOT tuned to it.")

print()
print(f"  {'auxiliary kW':>13} {'net kWh':>10} {'share of net':>14}")
print("  " + "-" * 40)
for aux in (0.0, 100.0, AUXILIARY_KW, 200.0):
    t3, _ = circuit_energy(results, traction, dwell_s=dwell, auxiliary_kw=aux)
    share = 100 * t3.auxiliary_j / t3.net_j if t3.net_j else 0.0
    print(f"  {aux:13.0f} {kwh(t3.net_j):10.1f} {share:13.1f}%")

# --------------------------------------------------------------- profile
print()
print("=" * 78)
print("CUMULATIVE ENERGY AROUND THE LOOP")
print("=" * 78)
print(f"  {'km':>6} {'traction kWh':>14} {'net kWh':>10}")
print("  " + "-" * 34)
marks = [0, 5000, 10000, 15000, 20000, 25000, 30000, 34500]
for m in marks:
    i = int(min(range(len(profile.distance_m)),
                key=lambda k: abs(profile.distance_m[k] - m)))
    print(f"  {profile.distance_m[i]/1000:6.1f} {kwh(profile.traction_j[i]):14.1f} "
          f"{kwh(profile.net_j[i]):10.1f}")
