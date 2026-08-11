"""Measure station dwell time from ODPT real-time train positions.

Usage:
    python scripts/measure_dwell.py                 # 3 minute sample
    python scripts/measure_dwell.py 600             # 10 minute sample
    python scripts/measure_dwell.py 600 --replay f  # re-analyse a saved sample

Set ODPT_CONSUMER_KEY to measure the Yamanote. Without a key this falls back to
the unauthenticated mirror, which carries Toei but no JR East -- useful for
exercising the pipeline, useless for this project's actual question.

TIMING MATTERS. Tokyo is UTC+9 and the Yamanote runs roughly 04:30-01:00 JST.
Sampling outside service hours produces episodes of parked trains, which are not
dwells. The script warns when it sees a feed that looks out of service.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from _env import ensure

ensure()

from tokyoline.odpt import (  # noqa: E402
    KEY_ENV,
    YAMANOTE,
    OdptClient,
    aggregate,
    collect,
    episodes_from_samples,
    load_samples,
    save_samples,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "dwell_samples.json"

args = [a for a in sys.argv[1:] if not a.startswith("--")]
duration = float(args[0]) if args else 180.0
replay = "--replay" in sys.argv

jst = timezone(timedelta(hours=9))
now_jst = datetime.now(jst)

print("=" * 78)
print("MEASURING STATION DWELL FROM ODPT REAL-TIME POSITIONS")
print("=" * 78)

client = OdptClient.from_env()
railway = YAMANOTE if client.authenticated else None

print(f"  endpoint       {client.base}")
print(f"  authenticated  {client.authenticated}")
print(f"  railway filter {railway or 'none (all operators on the public mirror)'}")
print(f"  Tokyo time     {now_jst:%Y-%m-%d %H:%M} JST")

if not client.authenticated:
    print()
    print("  NO CONSUMER KEY. The unauthenticated mirror carries Toei only --")
    print("  no JR East, so the Yamanote cannot be measured. What follows")
    print("  exercises the measurement pipeline against another operator's live")
    print("  data, which validates the method but answers a different question.")
    print(f"  Set {KEY_ENV} to measure the Yamanote itself.")

in_service = 5 <= now_jst.hour < 24 or now_jst.hour == 0
if not in_service:
    print()
    print(f"  WARNING: {now_jst:%H:%M} JST is outside normal service hours.")
    print("  Trains standing at platforms overnight are parked, not dwelling.")
    print("  Any 'dwell' measured now is meaningless.")

if replay and OUT.exists():
    print(f"\n  replaying {OUT}")
    samples = load_samples(OUT)
else:
    print(f"\n  sampling for {duration:.0f} s at 10 s intervals ...")
    samples = collect(client, railway=railway, duration_s=duration, interval_s=10.0)
    if samples:
        save_samples(samples, OUT)
        print(f"  saved {len(samples)} observations to {OUT}")

if not samples:
    print("\n  No observations collected. Nothing to analyse.")
    raise SystemExit(1)

episodes = episodes_from_samples(samples)
bracketed = [e for e in episodes if e.bracketed]

print()
print("=" * 78)
print("EPISODES")
print("=" * 78)
print(f"  observations         {len(samples)}")
print(f"  distinct trains      {len({s.train for s in samples})}")
print(f"  standing episodes    {len(episodes)}")
print(f"  fully bracketed      {len(bracketed)}")
print()
print("  An episode counts only if it is bracketed by running observations on")
print("  BOTH sides. One that was already in progress when sampling started, or")
print("  still in progress when it ended, is censored open-ended and is excluded")
print("  rather than counted as a short dwell.")

if not bracketed:
    print()
    print("  No bracketed episodes. Either the sample was too short to catch a")
    print("  train both arriving and departing, or nothing is running.")
    print("  Try a longer duration during service hours.")
    raise SystemExit(0)

stats = aggregate(episodes)

print()
print("=" * 78)
print("DWELL, AS AN INTERVAL")
print("=" * 78)
print("  Polling censors every measurement: a dwell is at least the observed")
print("  standing span, and at most the gap between the bracketing running")
print("  observations. Both bounds are reported. The midpoint is a convenience,")
print("  not a measurement.")
print()
print(f"  {stats.summary()}")

print()
print(f"  {'station':<28} {'n':>4} {'lower':>9} {'upper':>9}")
print("  " + "-" * 54)
for st, (n, lo, hi) in sorted(stats.per_station.items(),
                              key=lambda kv: -kv[1][0])[:20]:
    print(f"  {st:<28} {n:4d} {lo:8.1f}s {hi:8.1f}s")

if client.authenticated:
    print()
    print("=" * 78)
    print("WHAT THIS DOES TO PHASE 3")
    print("=" * 78)
    print("  Phase 3 inferred mean dwell of 39-50 s by constraining recovery")
    print("  margin to documented practice, because dwell could not be measured.")
    print(f"  Measured here: [{stats.mean_lower_s:.1f}, {stats.mean_upper_s:.1f}] s.")
    print()
    print("  If those overlap, the inference is corroborated by an independent")
    print("  route and the identifiability problem is genuinely closed: with")
    print("  dwell measured, recovery margin follows directly from the circuit")
    print("  equation rather than being assumed.")
    print()
    print("  Feed the midpoint into the dwell slider in the GUI, or into")
    print("  data/operational_practice.json, to propagate it.")
