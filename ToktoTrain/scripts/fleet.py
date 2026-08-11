"""Phase 4: fleet comparison across commuter EMU and high-speed classes."""

from __future__ import annotations

from _env import ensure

ensure()

from dataclasses import replace  # noqa: E402

from tokyoline.brake import DEFAULT_JERK_LIMIT  # noqa: E402
from tokyoline.fleet import Bounded, load_fleet  # noqa: E402

fleet = load_fleet()
JERK = DEFAULT_JERK_LIMIT

print("=" * 86)
print("PHASE 4  --  FLEET COMPARISON")
print("=" * 86)
print(f"  Shared comfort jerk limit: {JERK} m/s^3 across every class.")
print()

print(f"  {'class':<28} {'form':>7} {'cars':>5} {'mass t':>9} {'power kW':>9} "
      f"{'v_max':>7} {'accel':>7}")
print("  " + "-" * 82)
for m in fleet.values():
    mass = f"<{m.tare_mass_bound_t:.0f}" if m.mass_is_bound else f"{m.tare_mass_t:.1f}"
    print(f"  {m.display_name:<28} {m.formation:>7} {m.n_cars:5d} {mass:>9} "
          f"{m.total_power_kw:9.0f} {m.max_operating_speed_kmh:6.0f}k "
          f"{m.starting_acceleration_kmh_s:6.2f}")

print()
print("=" * 86)
print("THE CORE TRADE-OFF  --  acceleration against maximum speed")
print("=" * 86)
print("  Both from published figures. No modelling, no assumptions.")
print()
print(f"  {'class':<28} {'accel km/h/s':>13} {'v_max km/h':>12} {'product':>9}")
print("  " + "-" * 66)
for m in sorted(fleet.values(), key=lambda x: -x.starting_acceleration_kmh_s):
    print(f"  {m.display_name:<28} {m.starting_acceleration_kmh_s:13.2f} "
          f"{m.max_operating_speed_kmh:12.0f} "
          f"{m.starting_acceleration_kmh_s * m.max_operating_speed_kmh:9.0f}")
print()
print("  The commuter EMUs sit top-left (high acceleration, low top speed) and the")
print("  high-speed sets bottom-right. The N700S breaks the pattern: it accelerates")
print("  at 2.6 km/h/s, roughly half again what the E5 manages, despite being the")
print("  heaviest set here. That is an observation from published figures. Why JR")
print("  Central specified it that way is not something this data establishes, so")
print("  no reason is asserted here.")

print()
print("=" * 86)
print("POWER TO WEIGHT")
print("=" * 86)
print(f"  {'class':<28} {'kW/t':>12} {'kW/passenger':>14} {'motored':>9}")
print("  " + "-" * 68)
for m in sorted(fleet.values(), key=lambda x: -float(x.power_to_weight_kw_per_t())):
    ptw = m.power_to_weight_kw_per_t()
    shown = f">{float(ptw):.1f}" if isinstance(ptw, Bounded) else f"{float(ptw):.1f}"
    print(f"  {m.display_name:<28} {shown:>12} "
          f"{m.total_power_kw / m.capacity:14.2f} {100*m.motored_fraction:8.0f}%")
print()
print("  N700S is shown as a LOWER bound: its formation mass is published only as")
print("  'under 700 t', so dividing by it can only understate power-to-weight.")

print()
print("=" * 86)
print("BRAKING FROM OWN CRUISE SPEED  (identical jerk limit)")
print("=" * 86)
print(f"  {'class':<28} {'from':>7} {'rate':>16} {'distance':>11} {'time':>8}")
print("  " + "-" * 76)
for m in fleet.values():
    d = m.stopping_distance_m(jerk=JERK)
    if d is None:
        print(f"  {m.display_name:<28} {m.max_operating_speed_kmh:6.0f}k "
              f"{'not published':>16} {'--':>11} {'--':>8}")
        continue
    t = m.stopping_time_s(jerk=JERK)
    if m.deceleration_curve:
        rate = f"{m.deceleration_at_kmh(m.max_operating_speed_kmh):.2f}-2.69 var"
    else:
        rate = f"{m.service_deceleration_kmh_s:.2f} const"
    flag = " (!)" if m.decel_is_ambiguous else ""
    print(f"  {m.display_name:<28} {m.max_operating_speed_kmh:6.0f}k {rate:>16} "
          f"{d:9.0f} m {t:7.1f} s{flag}")

print()
print("  (!) E233 deceleration was published without a service/emergency qualifier.")
print("      Not a like-for-like comparison against the E235's explicit service rate.")
print()
print("  E5 and N700S publish no service deceleration in the sources consulted, so")
print("  they are omitted rather than filled with a peer value.")

print()
print("=" * 86)
print("WHY CONSTANT-DECELERATION MODELS FAIL FOR HIGH-SPEED STOCK")
print("=" * 86)
e7 = fleet["E7"]
print("  The E7 publishes a deceleration that varies with speed:")
for v, a in e7.deceleration_curve:
    print(f"    {v:5.0f} km/h  ->  {a:.2f} km/h/s")
print()
print("  Regenerative brake effort is power-limited exactly as tractive effort is,")
print("  so the achievable rate falls as speed rises. Modelling the E7 at its")
print("  low-speed rate throughout would give:")

const_d = e7.stopping_distance_m(jerk=JERK)
naive = replace(e7, deceleration_curve=None, service_deceleration_kmh_s=2.69)
naive_d = naive.stopping_distance_m(jerk=JERK)
print()
print(f"    correct (speed-dependent rate)  {const_d:8.0f} m")
print(f"    naive   (constant 2.69 km/h/s)  {naive_d:8.0f} m")
print(f"    understated by                  {const_d - naive_d:8.0f} m "
      f"({100 * (const_d - naive_d) / const_d:.0f}%)")
print()
print("  For the E235 at 90 km/h the constant assumption is harmless -- the whole")
print("  speed range sits in the flat part of the curve. For high-speed stock it")
print("  is not. That difference is the reason this project's Yamanote model can")
print("  use a constant service rate without apology, and a Shinkansen extension")
print("  could not.")

print()
print("=" * 86)
print("STEP-SIZE CHECK on the variable-rate integration")
print("=" * 86)
for dt in (0.05, 0.02, 0.01, 0.005, 0.002):
    print(f"    dt = {dt:6.3f} s  ->  {e7.stopping_distance_m(jerk=JERK, dt=dt):9.2f} m")
