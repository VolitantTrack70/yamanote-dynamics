"""Sanity checks on the physics core before any validation is attempted.

Everything printed here should be checked against physical intuition or a
published figure. If any of it looks wrong, the validation result downstream is
meaningless.
"""

from __future__ import annotations

from _env import ensure

ensure()

from tokyoline import build_model, load_segments, simulate_segment  # noqa: E402
from tokyoline.units import kmh  # noqa: E402

traction, brake = build_model(load_factor=1.0)
spec = traction.spec
davis = traction.davis

print("=" * 72)
print("ROLLING STOCK")
print("=" * 72)
print(f"  {spec.name}")
print(f"  formation           {spec.n_motor_cars}M{spec.n_cars - spec.n_motor_cars}T")
print(f"  tare mass           {spec.tare_mass_kg / 1000:.1f} t")
print(f"  loaded mass (AW2)   {spec.mass_kg(1.0) / 1000:.1f} t")
print(f"  effective mass      {spec.effective_mass_kg(1.0) / 1000:.1f} t  "
      f"(lambda = {spec.lambda_rot})")
print(f"  frontal area        {spec.frontal_area_m2:.2f} m^2")
print(f"  train length        {spec.train_length_m:.1f} m")

print()
print("=" * 72)
print("RUNNING RESISTANCE  (MODELLED -- no published E235 coefficients)")
print("=" * 72)
print(f"  A = {davis.A:9.1f} N        B = {davis.B:7.2f} N/(m/s)"
      f"        C = {davis.C:6.2f} N/(m/s)^2")
m = spec.mass_kg(1.0)
for v_kmh in (0, 30, 60, 90):
    v = v_kmh / 3.6
    print(f"  {v_kmh:3d} km/h   R = {davis(v):8.1f} N   "
          f"= {davis.specific_n_per_tonne(v, m):5.2f} N/t")
print("  expected range for a commuter EMU: ~10-15 N/t at rest, ~25-35 N/t at 90 km/h")

print()
print("=" * 72)
print("TRACTIVE EFFORT")
print("=" * 72)
print(f"  max effort          {traction.max_effort_n / 1000:.1f} kN")
print(f"  power (1-hour)      {traction.power_w / 1000:.0f} kW")
print(f"  base speed          {kmh(traction.base_speed_ms):.1f} km/h")
print(f"  max operating speed {kmh(spec.max_speed_ms):.1f} km/h")
print()
print("  speed   effort    adhesion ceiling   binding?")
for v_kmh in (0, 15, 30, 45, 60, 75, 90):
    v = v_kmh / 3.6
    F = traction.effort_n(v)
    F_adh = spec.adhesion_limit_n(v, 1.0)
    print(f"  {v_kmh:3d}     {F / 1000:6.1f} kN   {F_adh / 1000:8.1f} kN"
          f"          {'ADHESION' if traction.adhesion_binds(v) else 'no'}")

print()
print("=" * 72)
print("JERK-LIMITED BRAKING")
print("=" * 72)
print(f"  a_max = {brake.a_max:.3f} m/s^2 ({brake.a_max * 3.6:.1f} km/h/s), "
      f"jerk = {brake.jerk:.2f} m/s^3")
print(f"  trapezoidal above   {kmh(brake.speed_threshold_for_full_braking):.1f} km/h")
print()
print("  from      stop dist   idealised   jerk penalty   stop time")
for v_kmh in (30, 60, 90):
    v = v_kmh / 3.6
    print(f"  {v_kmh:3d} km/h   {brake.distance_to_stop(v):7.1f} m   "
          f"{brake.idealised_distance_to_stop(v):7.1f} m   "
          f"{brake.jerk_penalty_distance(v):8.1f} m   "
          f"{brake.time_to_stop(v):7.2f} s")
print(f"  time penalty per stop = a_max/jerk = {brake.a_max / brake.jerk:.2f} s")

print()
print("=" * 72)
print("ALL 30 SEGMENTS  (minimum run time, no coasting, no margin)")
print("=" * 72)
segments = load_segments()
print(f"  loaded {len(segments)} segments, "
      f"total {sum(s.distance_m for s in segments) / 1000:.1f} km")
print()

total = 0.0
worst_error = 0.0
n_cruise = 0
cruise_dist = 0.0
for seg in segments:
    r = simulate_segment(traction, brake, seg.distance_m,
                         speed_limit_ms=spec.max_speed_ms, dt=0.05)
    total += r.run_time_s
    worst_error = max(worst_error, abs(r.position_error_m))
    if r.reached_line_speed:
        n_cruise += 1
        cruise_dist += r.phase_distances_m["cruise"]
    print(f"  {seg.from_station:>18s} -> {seg.to_station:<18s} "
          f"{seg.distance_m:6.0f} m  {r.run_time_s:6.1f} s  "
          f"peak {r.max_speed_kmh:5.1f} km/h  "
          f"cruise {r.phase_distances_m['cruise']:6.0f} m")

print()
print(f"  sum of run times          {total:.1f} s  ({total / 60:.2f} min)")
print(f"  segments reaching 90 km/h {n_cruise}/{len(segments)}")
print(f"  total distance at cruise  {cruise_dist / 1000:.2f} km "
      f"({100 * cruise_dist / sum(s.distance_m for s in segments):.0f}% of the loop)")
print(f"  worst platform error      {worst_error:.2e} m")
print()
print("  published loop time (mean, inner) 3948 s = 65.80 min")
print(f"  implied total dwell + margin      {3948 - total:.0f} s "
      f"= {(3948 - total) / len(segments):.1f} s per station")
