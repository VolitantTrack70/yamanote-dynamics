"""Loading the data layer and assembling a default model.

Distances come from JR East's published operating kilometrage (営業キロ), NOT
from great-circle distance between station coordinates. That choice matters: the
Yamanote is a curvy loop, so straight-line distance between consecutive stations
understates track distance, and correcting it with a single global scale factor
to hit 34.5 km smears the residual unevenly across segments. That residual would
land directly on top of the systematic bias this project exists to measure.
Published kilometrage removes the error source instead of correcting for it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .brake import DEFAULT_JERK_LIMIT, BrakeProfile
from .resistance import DavisCoefficients
from .stock import TrainSpec
from .traction import TractionModel

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

#: Published total loop length, km. Used as an integrity check on the CSV.
PUBLISHED_LOOP_KM = 34.5


@dataclass(frozen=True)
class Segment:
    """One interstation run."""

    index: int
    from_station: str
    to_station: str
    distance_m: float

    grade_permille: float = 0.0
    """Average gradient, positive uphill in the outer-loop direction.

    Zero unless segments are loaded with `gradients=True`. See
    :func:`load_gradients` for why the default is zero and why the derived
    profile is a bounding study rather than an input to the headline results.
    """

    def __str__(self) -> str:
        return f"{self.from_station} -> {self.to_station} ({self.distance_m:.0f} m)"

    def reversed_grade(self) -> "Segment":
        """The same segment travelled the other way round the loop."""
        return Segment(self.index, self.to_station, self.from_station,
                       self.distance_m, -self.grade_permille)


def load_stations(path: Path | str | None = None) -> pd.DataFrame:
    """Load the station table, verifying it closes the loop."""
    path = Path(path) if path else DATA_DIR / "yamanote_stations.csv"
    df = pd.read_csv(path, comment="#")

    total = df["dist_from_prev_km"].sum()
    if abs(total - PUBLISHED_LOOP_KM) > 0.05:
        raise ValueError(
            f"station distances sum to {total:.2f} km, "
            f"but the published loop is {PUBLISHED_LOOP_KM} km"
        )
    return df


def load_gradients(path: Path | str | None = None) -> dict[tuple[str, str], float]:
    """Average per-segment gradient in per-mille, DERIVED from terrain elevation.

    NOT used by any headline result, and that is deliberate. The values come from
    ground elevation at each station coordinate, and railways are graded
    precisely so that track does NOT follow the ground: viaducts and cuttings
    hold the rails level while the terrain moves. Ground elevation therefore
    systematically OVERSTATES track gradient, and feeding it into the physics
    would inject a new error while appearing to remove one.

    Its legitimate use is as a bound. Running the model with gradients this large
    puts a ceiling on how much the flat-track assumption can be costing, which is
    better than asserting the assumption is harmless. See scripts/gradient.py.
    """
    path = Path(path) if path else DATA_DIR / "gradients.csv"
    df = pd.read_csv(path, comment="#")
    return {
        (r.from_station, r.to_station): float(r.grade_permille)
        for r in df.itertuples()
    }


def load_segments(path: Path | str | None = None, *,
                  gradients: bool = False) -> list[Segment]:
    """Build the 30 interstation segments in outer-loop order.

    Segment i runs from station i to station i+1, and the final segment closes
    the loop back to the origin. There are exactly as many segments as stations
    because it is a loop, which is worth stating because it is the one place an
    off-by-one would silently drop 0.9 km.
    """
    df = load_stations(path)
    names = df["name_en"].tolist()
    n = len(names)

    grades = load_gradients() if gradients else {}

    segments: list[Segment] = []
    for i in range(n):
        to_idx = (i + 1) % n
        # dist_from_prev_km on the DESTINATION row is the length of this segment.
        distance_km = float(df.loc[to_idx, "dist_from_prev_km"])
        segments.append(
            Segment(
                index=i,
                from_station=names[i],
                to_station=names[to_idx],
                distance_m=distance_km * 1000.0,
                grade_permille=grades.get((names[i], names[to_idx]), 0.0),
            )
        )
    return segments


def load_ridership(path: Path | str | None = None) -> pd.DataFrame:
    """JR East published daily BOARDINGS per station, FY2024.

    Boardings only. Not boardings plus alightings, and not segment loading.
    Turning this into train occupancy would require an origin-destination matrix
    that is not public. See the header of the CSV.
    """
    path = Path(path) if path else DATA_DIR / "ridership.csv"
    return pd.read_csv(path, comment="#")


def load_congestion(path: Path | str | None = None) -> dict:
    """MLIT congestion rates for the Yamanote's critical sections.

    Covers one section per direction at peak hour. That is the whole of the
    public segment-level loading data for this line.
    """
    path = Path(path) if path else DATA_DIR / "congestion.json"
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_spec(path: Path | str | None = None, key: str = "E235-0") -> TrainSpec:
    path = Path(path) if path else DATA_DIR / "rolling_stock.json"
    return TrainSpec.from_json(path, key=key)


def build_model(
    spec: TrainSpec | None = None,
    *,
    load_factor: float = 1.0,
    jerk_limit: float = DEFAULT_JERK_LIMIT,
    power_factor: float = 1.0,
) -> tuple[TractionModel, BrakeProfile]:
    """Assemble the default traction and braking models.

    `load_factor` of 1.0 is nominal full capacity. The Yamanote routinely
    exceeds this at peak -- published congestion rates are above 100 percent --
    so this is a mid-range operating assumption, not a worst case.
    """
    spec = spec or load_spec()

    davis = DavisCoefficients.estimate_for_emu(
        mass_kg=spec.mass_kg(load_factor),
        n_cars=spec.n_cars,
        frontal_area_m2=spec.frontal_area_m2,
    )
    traction = TractionModel(
        spec=spec, davis=davis, load_factor=load_factor, power_factor=power_factor
    )
    brake = BrakeProfile(a_max=spec.service_decel_ms2, jerk=jerk_limit)
    return traction, brake
