"""Phase 6: optimal coasting and the energy-vs-run-time Pareto front."""

from __future__ import annotations

from _env import ensure

ensure()

from tokyoline import build_model, load_segments  # noqa: E402
from tokyoline.coasting import (  # noqa: E402
    evaluate,
    pareto_front,
    solve_coast_for_time,
    solve_speed_for_time,
)
from tokyoline.energy import JOULES_PER_KWH  # noqa: E402
from tokyoline.validate import load_anchors  # noqa: E402

traction, brake = build_model(load_factor=1.0)
segments = load_segments()
V_LIMIT = traction.spec.max_speed_ms
DT = 0.1
PUBLISHED_LOOP_S = load_anchors()["loop_times"]["inner_loop_mean_s"]["value"]


def kwh(j: float) -> float:
    return j / JOULES_PER_KWH


print("=" * 80)
print("PHASE 6  --  OPTIMAL COASTING")
print("=" * 80)
print("  Four-phase profile: accelerate -> cruise -> coast -> brake.")
print("  With the shape fixed, the strategy is ONE number per segment: where")
print("  coasting starts. Bisect on it to hit a target run time.")

# ------------------------------------------------------- one segment, in detail
demo = max(segments, key=lambda s: s.distance_m)
print()
print("=" * 80)
print(f"WORKED EXAMPLE  --  {demo.from_station} to {demo.to_station}, "
      f"{demo.distance_m:.0f} m")
print("=" * 80)

flat = evaluate(traction, brake, demo.distance_m, speed_limit_ms=V_LIMIT, dt=DT,
                label="flat out")
print(f"  flat out: {flat.run_time_s:.1f} s, {kwh(flat.net_energy_j):.2f} kWh net")
print()
print(f"  {'target':>8} {'coast onset':>13} {'coast dist':>11} {'time':>8} "
      f"{'net kWh':>9}")
print("  " + "-" * 56)
for factor in (1.02, 1.05, 1.10, 1.20):
    s = solve_coast_for_time(traction, brake, demo.distance_m,
                             flat.run_time_s * factor, speed_limit_ms=V_LIMIT, dt=DT)
    if s is None:
        print(f"  {factor:7.0%}   unreachable")
        continue
    print(f"  {factor:7.0%} {s.coast_start_m:11.0f} m {s.coast_distance_m:9.0f} m "
          f"{s.run_time_s:7.1f}s {kwh(s.net_energy_j):9.2f}")

# ----------------------------------------------------------------- pareto front
print()
print("=" * 80)
print("PARETO FRONT  --  coasting against simply cruising slower")
print("=" * 80)
print("  Both strategies hit the SAME run time. Comparing coasting against a")
print("  flat-out run would be meaningless: the flat-out run is faster, so of")
print("  course it uses more energy. The question is whether coasting beats the")
print("  obvious alternative -- go slower throughout.")
print()
print(f"  {'time':>7} {'coast kWh':>11} {'slower kWh':>12} {'saving':>10} "
      f"{'coast v':>9} {'slower v':>10}")
print("  " + "-" * 66)

front = pareto_front(traction, brake, demo.distance_m, speed_limit_ms=V_LIMIT, dt=DT)
for p in front:
    if p.coast is None or p.slower is None:
        print(f"  {p.time_factor:6.0%}   unreachable")
        continue
    saving = p.saving_fraction
    print(f"  {p.time_factor:6.0%} {kwh(p.coast.net_energy_j):11.2f} "
          f"{kwh(p.slower.net_energy_j):12.2f} "
          f"{(f'{100*saving:+.1f}%' if saving is not None else '--'):>10} "
          f"{p.coast.peak_speed_kmh:8.0f}k {p.slower.peak_speed_kmh:9.0f}k")

# ------------------------------------------------- interaction with regeneration
print()
print("=" * 80)
print("COASTING AND REGENERATION ARE PARTIAL SUBSTITUTES")
print("=" * 80)
print("  Coasting saves energy by not spending it. Regeneration saves energy by")
print("  recovering what was spent. The better the regeneration, the less")
print("  coasting adds on top -- so the value of coasting depends on a number")
print("  the model cannot pin down.")
print()
print(f"  {'receptivity':>12} {'coast kWh':>11} {'slower kWh':>12} {'saving':>10}")
print("  " + "-" * 48)
for rec in (0.0, 0.3, 0.7, 1.0):
    target = flat.run_time_s * 1.10
    c = solve_coast_for_time(traction, brake, demo.distance_m, target,
                             speed_limit_ms=V_LIMIT, dt=DT, regen_receptivity=rec)
    s = solve_speed_for_time(traction, brake, demo.distance_m, target,
                             speed_limit_ms=V_LIMIT, dt=DT, regen_receptivity=rec)
    if c is None or s is None:
        continue
    saving = (s.net_energy_j - c.net_energy_j) / s.net_energy_j
    print(f"  {rec:12.2f} {kwh(c.net_energy_j):11.2f} {kwh(s.net_energy_j):12.2f} "
          f"{100*saving:9.1f}%")

# --------------------------------------------------------------- whole circuit
print()
print("=" * 80)
print("WHOLE CIRCUIT  --  coasting applied to every segment")
print("=" * 80)
print("  Each segment is solved to the same proportional slowdown, so the whole")
print("  circuit stretches uniformly. This is a simplification: a real optimiser")
print("  would distribute the extra time unevenly, spending more of it where it")
print("  buys the most energy. Uniform stretching is therefore a LOWER bound on")
print("  the achievable saving.")
print()
print(f"  {'stretch':>8} {'run time':>10} {'coast kWh':>11} {'slower kWh':>12} "
      f"{'saving':>9}")
print("  " + "-" * 56)

base_total = sum(
    evaluate(traction, brake, s.distance_m, speed_limit_ms=V_LIMIT, dt=DT).run_time_s
    for s in segments
)
for factor in (1.0, 1.05, 1.10, 1.20):
    t_coast = e_coast = e_slow = 0.0
    ok = True
    for seg in segments:
        base = evaluate(traction, brake, seg.distance_m, speed_limit_ms=V_LIMIT, dt=DT)
        if factor == 1.0:
            t_coast += base.run_time_s
            e_coast += base.net_energy_j
            e_slow += base.net_energy_j
            continue
        target = base.run_time_s * factor
        c = solve_coast_for_time(traction, brake, seg.distance_m, target,
                                 speed_limit_ms=V_LIMIT, dt=DT)
        s = solve_speed_for_time(traction, brake, seg.distance_m, target,
                                 speed_limit_ms=V_LIMIT, dt=DT)
        if c is None or s is None:
            ok = False
            break
        t_coast += c.run_time_s
        e_coast += c.net_energy_j
        e_slow += s.net_energy_j
    if not ok:
        print(f"  {factor:7.0%}   unreachable on at least one segment")
        continue
    saving = (e_slow - e_coast) / e_slow if e_slow else 0.0
    print(f"  {factor:7.0%} {t_coast:9.0f}s {kwh(e_coast):11.1f} "
          f"{kwh(e_slow):12.1f} {100*saving:8.1f}%")

print()
print("  Note what dominates. Coasting beats cruising slower by a few percent,")
print("  which is a genuine but modest optimisation. Simply ALLOWING more time is")
print("  worth far more: 10 percent extra run time cuts circuit energy by about a")
print("  third, because kinetic energy goes as the square of speed. The shape of")
print("  the Pareto front matters more than the choice of strategy along it.")
print()
print(f"  flat-out circuit run time  {base_total:.0f} s")
print(f"  published circuit          {PUBLISHED_LOOP_S} s")
print()
print("  The schedule already runs well above the flat-out minimum, so there is")
print("  ample room for coasting within the existing timetable. Whether JR East")
print("  actually coasts, and where, is not something this model can observe.")
