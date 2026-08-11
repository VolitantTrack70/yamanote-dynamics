"""Step-size sensitivity study for the fixed-step RK4 integrator.

Evidence for the README that the integrator has converged. Without this, every
run-time figure in the project is an unquantified claim.
"""

from __future__ import annotations

from _env import ensure

ensure()

from tokyoline import build_model, load_segments, simulate_segment  # noqa: E402

traction, brake = build_model(load_factor=1.0)
segments = load_segments()
v_limit = traction.spec.max_speed_ms

STEPS = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]

print("Loop total run time vs integration step")
print()
print("    dt (s)     loop run time (s)      change vs finer      worst platform err")
print("  " + "-" * 76)

results = []
for dt in STEPS:
    total = 0.0
    worst = 0.0
    for seg in segments:
        r = simulate_segment(traction, brake, seg.distance_m,
                             speed_limit_ms=v_limit, dt=dt)
        total += r.run_time_s
        worst = max(worst, abs(r.position_error_m))
    results.append((dt, total, worst))

reference = results[-1][1]
for i, (dt, total, worst) in enumerate(results):
    delta = total - results[i + 1][1] if i + 1 < len(results) else 0.0
    print(f"  {dt:8.3f}   {total:14.4f}       {delta:+12.5f} s      {worst:.2e} m")

print()
print(f"  reference (dt = {STEPS[-1]} s): {reference:.4f} s")
for dt, total, _ in results:
    err = total - reference
    print(f"  dt = {dt:6.3f} s  ->  error {err:+9.5f} s  "
          f"({100 * abs(err) / reference:.5f}% of loop run time)")

print()
print("  RK4 is 4th order, so halving dt should cut the error by roughly 16x")
print("  in the smooth regions. Phase-switch handling limits the observed order.")
