"""Phase 3: the centerpiece. Simulated minimum run time vs the published timetable."""

from __future__ import annotations

from _env import ensure

ensure()

from tokyoline import BrakeProfile, build_model, load_segments  # noqa: E402
from tokyoline.validate import (  # noqa: E402
    decompose,
    dwell_implied_by_margin,
    dwell_implied_by_zero_margin,
    identifiability_locus,
    load_anchors,
    load_practice,
    simulate_loop,
)

traction, brake = build_model(load_factor=1.0)
segments = load_segments()
practice = load_practice()
anchors = load_anchors()

PUBLISHED_LOOP_S = anchors["loop_times"]["inner_loop_mean_s"]["value"]
FASTEST_LOOP_S = anchors["loop_times"]["fastest_circuit_s"]["value"]
LINE_SPEED_KMH = anchors["line_speed_limit_kmh"]["value"]
DWELL = practice["yamanote_dwell_scenarios_seconds"]

print("=" * 78)
print("PHASE 3  --  SIMULATED MINIMUM RUN TIME vs PUBLISHED TIMETABLE")
print("=" * 78)
print(f"  line               Yamanote, {len(segments)} segments, "
      f"{sum(s.distance_m for s in segments) / 1000:.1f} km")
print(f"  stock              {traction.spec.name}")
print(f"  published loop     {PUBLISHED_LOOP_S} s  (inner loop, mean of all services)")
print(f"  fastest circuit    {FASTEST_LOOP_S} s")

minimum = simulate_loop(traction, brake, segments, label="minimum",
                        speed_limit_kmh=LINE_SPEED_KMH)
print(f"  simulated minimum  {minimum.raw_total_s:.0f} s")
print(f"  raw gap            {PUBLISHED_LOOP_S - minimum.raw_total_s:.0f} s "
      f"({100 * (PUBLISHED_LOOP_S - minimum.raw_total_s) / PUBLISHED_LOOP_S:.0f}% "
      f"of the published loop)")
print()
print("  A raw gap of this size is NOT a recovery-margin measurement. Almost all")
print("  of it is dwell and documented de-rating practice. Decompose before")
print("  attributing anything.")

print()
print("=" * 78)
print("DECOMPOSITION LADDER  (central dwell scenario)")
print("=" * 78)
central = decompose(traction, brake, segments,
                    dwell_s=DWELL["central"], published_loop_s=PUBLISHED_LOOP_S,
                    line_speed_kmh=LINE_SPEED_KMH)
print(central.table())
print()
print(f"  standard run time (基準運転時分)   {central.standard_run_time_s:.0f} s")
print(f"  residual margin                    {central.residual_margin_s:.0f} s "
      f"= {100 * central.residual_margin_fraction:.1f}% of standard run time")

print()
print("=" * 78)
print("DWELL SENSITIVITY  --  the assumption the result hangs on")
print("=" * 78)
print(f"  {'scenario':<10} {'dwell':>8} {'total dwell':>13} {'residual':>11} "
      f"{'margin':>10}")
print("  " + "-" * 60)
for name in ("low", "central", "high"):
    d = decompose(traction, brake, segments, dwell_s=DWELL[name],
                  published_loop_s=PUBLISHED_LOOP_S, line_speed_kmh=LINE_SPEED_KMH)
    print(f"  {name:<10} {DWELL[name]:6.0f} s {DWELL[name] * len(segments):11.0f} s "
          f"{d.residual_margin_s:9.0f} s {100 * d.residual_margin_fraction:9.1f}%")

zero_margin_dwell = dwell_implied_by_zero_margin(
    central.standard_run_time_s, PUBLISHED_LOOP_S, len(segments))
print()
print(f"  dwell implying ZERO margin: {zero_margin_dwell:.1f} s per station")
print("  Documented JR East practice explicitly adds margin, so true mean dwell")
print("  must be below this. It is a hard upper bound on dwell, and therefore a")
print("  hard lower bound on margin.")

print()
print("=" * 78)
print("IDENTIFIABILITY LOCUS")
print("=" * 78)
print("  Every (dwell, margin) pair below fits the published loop time exactly.")
print("  Public data cannot choose between them. This is the honest result.")
print()
locus = identifiability_locus(central.standard_run_time_s, PUBLISHED_LOOP_S,
                              len(segments), dwell_range_s=(15.0, 50.0), n_points=8)
print(f"  {'mean dwell (s)':>16} {'implied margin':>16}")
print("  " + "-" * 34)
for d, m in zip(locus["dwell_s"], locus["margin_fraction"]):
    flag = "  <-- exceeds documented practice" if m > 0.10 else ""
    print(f"  {d:14.1f}   {100 * m:14.1f}%{flag}")

print()
print("=" * 78)
print("DERATING SENSITIVITY  (MLIT consensus range is 2-5 km/h)")
print("=" * 78)
print(f"  {'derate':>8} {'standard run':>14} {'residual @ central dwell':>26}")
print("  " + "-" * 52)
for derate in (0.0, 2.0, 3.0, 5.0):
    d = decompose(traction, brake, segments, dwell_s=DWELL["central"],
                  published_loop_s=PUBLISHED_LOOP_S, line_speed_kmh=LINE_SPEED_KMH,
                  derate_kmh=derate)
    print(f"  {derate:5.0f} km/h {d.standard_run_time_s:12.0f} s "
          f"{d.residual_margin_s:16.0f} s  ({100 * d.residual_margin_fraction:.1f}%)")

print()
print("=" * 78)
print("BRAKE RATE SENSITIVITY  --  the largest single lever in the model")
print("=" * 78)
print("  4.2 km/h/s is the MAXIMUM service brake rate. Drivers do not use maximum")
print("  service braking into every platform, and MLIT records that operators")
print("  build driveability into the run curve. Nothing in the public record says")
print("  what rate the Yamanote standard run time assumes.")
print()
print(f"  {'brake rate':>12} {'stop from 90':>14} {'standard run':>14} "
      f"{'residual @ 32s dwell':>22}")
print("  " + "-" * 66)
for rate_kmh_s in (4.2, 3.5, 3.0, 2.5):
    b = BrakeProfile(a_max=rate_kmh_s / 3.6, jerk=brake.jerk)
    d = decompose(traction, b, segments, dwell_s=DWELL["central"],
                  published_loop_s=PUBLISHED_LOOP_S, line_speed_kmh=LINE_SPEED_KMH)
    print(f"  {rate_kmh_s:9.1f} km/h/s {b.distance_to_stop(25.0):11.0f} m "
          f"{d.standard_run_time_s:12.0f} s {d.residual_margin_s:16.0f} s "
          f"({100 * d.residual_margin_fraction:.1f}%)")

print()
print("=" * 78)
print("INVERTED INFERENCE  --  constrain the DOCUMENTED quantity, solve for the")
print("                        UNDOCUMENTED one")
print("=" * 78)
print("  Margin practice is documented (MLIT: JR East adds margin; industry norm")
print("  is a few percent). Dwell is not documented for this line at all. So fix")
print("  the margin at documented levels and solve for the dwell the timetable")
print("  implies. This runs the inference in the direction the evidence supports,")
print("  and avoids tuning a dwell assumption until the margin looks right.")
print()
for rate_kmh_s in (4.2, 3.0):
    b = BrakeProfile(a_max=rate_kmh_s / 3.6, jerk=brake.jerk)
    d = decompose(traction, b, segments, dwell_s=DWELL["central"],
                  published_loop_s=PUBLISHED_LOOP_S, line_speed_kmh=LINE_SPEED_KMH)
    print(f"  brake rate {rate_kmh_s} km/h/s, standard run {d.standard_run_time_s:.0f} s")
    print(f"    {'assumed margin':>16} {'implied mean dwell':>20}")
    for margin in (0.03, 0.05, 0.08, 0.10):
        dw = dwell_implied_by_margin(d.standard_run_time_s, PUBLISHED_LOOP_S,
                                     len(segments), margin)
        print(f"    {100 * margin:14.0f}% {dw:18.1f} s")
    print()

print("=" * 78)
print("PER-SEGMENT  (quantization-limited -- see warning)")
print("=" * 78)
print("  Public per-segment times are minute-rounded. A 2-minute segment carries")
print("  +/- 30 s of quantization against a few-percent effect. Reported for")
print("  completeness; not used for any quantitative claim.")
print()
rounded = simulate_loop(traction, brake, segments, label="standard",
                        speed_limit_kmh=LINE_SPEED_KMH - 3.0, round_up_to_s=5.0)
print(f"  {'segment':<42} {'dist':>7} {'sim':>7} {'+dwell':>8}")
print("  " + "-" * 68)
for r in rounded.runs:
    label = f"{r.segment.from_station} -> {r.segment.to_station}"
    print(f"  {label:<42} {r.segment.distance_m:6.0f}m {r.run_time_s:6.1f}s "
          f"{r.run_time_s + DWELL['central']:7.1f}s")
