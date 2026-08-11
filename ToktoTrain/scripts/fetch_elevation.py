"""Fetch station elevations and derive a per-segment gradient profile.

The project has always modelled grade resistance and then evaluated it at zero,
on the grounds that per-segment gradient data for the Yamanote is not published.
That is true of *track* gradient. But station elevations can be sampled from a
public terrain model, and combined with the published kilometrage that gives an
average gradient per segment.

What this is NOT:

  * It is not track gradient. It is ground elevation at the station coordinate.
    Where the Yamanote runs on viaduct or in cutting -- which is much of it --
    the rails are not at ground level, and several stations (Shibuya, Ueno) have
    platforms well above or below the surrounding ground.
  * It is an AVERAGE over the segment, not a maximum. Real track has local
    grades steeper than the segment average.

So this is a derived, indicative profile, not a survey. Its value is that it can
be cross-checked: ja.wikipedia gives the line's maximum gradient as 34 permille
between Tabata and Nishi-Nippori, so the derived profile should show that segment
as by far the steepest, with an average somewhat below 34. If it does, the
profile is picking up real terrain. If it does not, it is noise and should be
discarded rather than used.

Writes data/gradients.csv. Run once; the output is committed.
"""

from __future__ import annotations

import csv
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from _env import ensure

ensure()

from tokyoline.data import DATA_DIR, load_segments  # noqa: E402
from tokyoline.network import load_coordinates  # noqa: E402

ENDPOINT = "https://api.open-elevation.com/api/v1/lookup"


def fetch_elevations(points: list[tuple[float, float]]) -> list[float] | None:
    """Query a public terrain model for elevations, in metres."""
    payload = json.dumps({
        "locations": [{"latitude": lat, "longitude": lon} for lat, lon in points]
    }).encode()
    req = urllib.request.Request(
        ENDPOINT, data=payload,
        headers={"Content-Type": "application/json",
                 "User-Agent": "tokyoline-model/0.1 (gradient profile)"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            return [float(r["elevation"]) for r in body["results"]]
        except (urllib.error.URLError, KeyError, ValueError) as exc:
            print(f"  attempt {attempt + 1} failed: {exc}", file=sys.stderr)
            time.sleep(3)
    return None


def main() -> int:
    coords = load_coordinates()
    segments = load_segments()
    names = [s.from_station for s in segments]
    by_name = {r.name_en: (r.lat, r.lon) for r in coords.itertuples()}

    points = [by_name[n] for n in names]
    print(f"querying elevations for {len(points)} stations ...", file=sys.stderr)
    elevations = fetch_elevations(points)
    if elevations is None:
        print("FAILED: could not reach the elevation service. Nothing written.",
              file=sys.stderr)
        return 1

    rows = []
    n = len(names)
    for i, seg in enumerate(segments):
        j = (i + 1) % n
        rise = elevations[j] - elevations[i]
        grade = 1000.0 * rise / seg.distance_m  # permille
        rows.append((
            seg.from_station, seg.to_station,
            round(elevations[i], 1), round(elevations[j], 1),
            round(rise, 1), round(seg.distance_m, 0), round(grade, 2),
        ))

    steepest = max(rows, key=lambda r: abs(r[6]))
    total_rise = sum(r[4] for r in rows)

    out = DATA_DIR / "gradients.csv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        fh.write("# Average per-segment gradient, DERIVED, not published.\n")
        fh.write("# elevation_* are ground elevations at the station coordinate\n")
        fh.write("# from a public terrain model (open-elevation.com), NOT track\n")
        fh.write("# level. grade_permille is the segment AVERAGE over the published\n")
        fh.write("# kilometrage; real track has steeper local grades.\n")
        fh.write("# Positive grade = uphill in the outer-loop direction.\n")
        w = csv.writer(fh)
        w.writerow(["from_station", "to_station", "elevation_from_m",
                    "elevation_to_m", "rise_m", "distance_m", "grade_permille"])
        w.writerows(rows)

    print(f"wrote {len(rows)} segments to {out}", file=sys.stderr)
    print(f"  steepest: {steepest[0]} -> {steepest[1]} at "
          f"{steepest[6]:+.1f} permille", file=sys.stderr)
    print(f"  elevation range: {min(elevations):.0f}-{max(elevations):.0f} m",
          file=sys.stderr)
    print(f"  net rise around the loop: {total_rise:+.1f} m "
          f"(must be ~0 -- it is a closed loop)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
