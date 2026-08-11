"""Fetch Yamanote station coordinates from OpenStreetMap via the Overpass API.

Queried programmatically rather than transcribed by hand or by a model: 30
latitude/longitude pairs copied by eye is a step that silently introduces
errors, and a wrong coordinate would show up as a kink in the line map that
looks like a rendering bug rather than a data bug.

Writes data/station_coordinates.csv. Run once; the output is committed.
"""

from __future__ import annotations

import csv
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

# Japanese names, in outer-loop order, matching yamanote_stations.csv.
STATIONS = [
    ("Shinagawa", "品川"), ("Osaki", "大崎"), ("Gotanda", "五反田"),
    ("Meguro", "目黒"), ("Ebisu", "恵比寿"), ("Shibuya", "渋谷"),
    ("Harajuku", "原宿"), ("Yoyogi", "代々木"), ("Shinjuku", "新宿"),
    ("Shin-Okubo", "新大久保"), ("Takadanobaba", "高田馬場"), ("Mejiro", "目白"),
    ("Ikebukuro", "池袋"), ("Otsuka", "大塚"), ("Sugamo", "巣鴨"),
    ("Komagome", "駒込"), ("Tabata", "田端"), ("Nishi-Nippori", "西日暮里"),
    ("Nippori", "日暮里"), ("Uguisudani", "鶯谷"), ("Ueno", "上野"),
    ("Okachimachi", "御徒町"), ("Akihabara", "秋葉原"), ("Kanda", "神田"),
    ("Tokyo", "東京"), ("Yurakucho", "有楽町"), ("Shimbashi", "新橋"),
    ("Hamamatsucho", "浜松町"), ("Tamachi", "田町"),
    ("Takanawa Gateway", "高輪ゲートウェイ"),
]

OVERPASS = "https://overpass-api.de/api/interpreter"
BBOX = "35.60,139.65,35.80,139.82"


def build_query() -> str:
    names = "|".join(ja for _, ja in STATIONS)
    return f"""
[out:json][timeout:120];
(
  node["railway"="station"]["name"~"^({names})$"]({BBOX});
  way["railway"="station"]["name"~"^({names})$"]({BBOX});
);
out center tags;
"""


def fetch(query: str) -> dict:
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(
        OVERPASS, data=data,
        headers={"User-Agent": "tokyoline-model/0.1 (station coordinate fetch)"},
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.loads(resp.read().decode("utf-8"))


def pick_best(elements: list[dict]) -> dict | None:
    """Prefer the JR East node when several operators share a station name.

    Tokyo Metro, Toei and JR nodes for the same interchange sit within ~100 m of
    each other. For a line map any of them would look fine, but picking
    deliberately keeps the choice documented rather than incidental.
    """
    if not elements:
        return None
    for el in elements:
        op = (el.get("tags", {}).get("operator", "")
              + el.get("tags", {}).get("operator:en", ""))
        if "東日本" in op or "JR East" in op or "JR" in op:
            return el
    return elements[0]


def main() -> int:
    print("querying Overpass ...", file=sys.stderr)
    try:
        result = fetch(build_query())
    except Exception as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    by_name: dict[str, list[dict]] = {}
    for el in result.get("elements", []):
        name = el.get("tags", {}).get("name")
        if name:
            by_name.setdefault(name, []).append(el)

    rows = []
    missing = []
    for en, ja in STATIONS:
        best = pick_best(by_name.get(ja, []))
        if best is None:
            missing.append(f"{en} ({ja})")
            continue
        lat = best.get("lat") or best.get("center", {}).get("lat")
        lon = best.get("lon") or best.get("center", {}).get("lon")
        if lat is None or lon is None:
            missing.append(f"{en} ({ja}) - no coordinate")
            continue
        operator = best.get("tags", {}).get("operator", "")
        rows.append((en, ja, round(float(lat), 6), round(float(lon), 6), operator))

    if missing:
        print(f"MISSING {len(missing)}: {', '.join(missing)}", file=sys.stderr)

    out = DATA_DIR / "station_coordinates.csv"
    with open(out, "w", encoding="utf-8", newline="") as fh:
        fh.write("# Station coordinates from OpenStreetMap via the Overpass API.\n")
        fh.write("# Fetched programmatically, not transcribed. Order matches\n")
        fh.write("# yamanote_stations.csv (outer loop from Shinagawa).\n")
        fh.write("# Used for the line map ONLY. Distances come from published\n")
        fh.write("# kilometrage, never from these coordinates -- see data.py.\n")
        w = csv.writer(fh)
        w.writerow(["name_en", "name_ja", "lat", "lon", "osm_operator"])
        w.writerows(rows)

    print(f"wrote {len(rows)}/{len(STATIONS)} stations to {out}", file=sys.stderr)
    return 0 if len(rows) == len(STATIONS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
