"""Rolling stock specification.

Loaded from data/rolling_stock.json, which tags every parameter as published,
derived or modelled. That tagging is not decoration -- the README depends on it,
and so does any honest reading of the validation result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .units import G, ms, ms2


@dataclass(frozen=True)
class TrainSpec:
    """Physical and performance parameters for one train class.

    All fields are SI. Conversion from the published km/h and km/h/s figures
    happens once, in :meth:`from_json`.
    """

    name: str
    n_cars: int
    n_motor_cars: int

    tare_mass_kg: float
    capacity: int
    passenger_mass_kg: float

    total_power_w: float
    """One-hour rating. See the note in :meth:`tractive_effort_limit`."""

    max_speed_ms: float
    start_accel_ms2: float
    service_decel_ms2: float

    lambda_rot: float
    """Rotational inertia allowance. m_eff = m * (1 + lambda)."""

    adhesive_mass_fraction: float
    body_width_m: float
    body_height_m: float
    car_length_m: float

    # ---------------------------------------------------------------- masses

    def mass_kg(self, load_factor: float = 0.0) -> float:
        """Static mass at a given load.

        `load_factor` is passengers as a fraction of published capacity.
        0.0 is tare (AW0), 1.0 is nominal full capacity (roughly AW2).
        Values above 1.0 are meaningful on the Yamanote at peak, where
        published congestion rates exceed 100 percent.
        """
        return self.tare_mass_kg + load_factor * self.capacity * self.passenger_mass_kg

    def effective_mass_kg(self, load_factor: float = 0.0) -> float:
        """Mass including the rotational inertia of wheels, axles and motors.

        Rotating components must be angularly accelerated as well as translated.
        Ignoring this makes the train roughly 8 percent too quick, which is the
        same order as the effect this project is trying to measure.
        """
        return self.mass_kg(load_factor) * (1.0 + self.lambda_rot)

    def adhesive_mass_kg(self, load_factor: float = 0.0) -> float:
        """Mass carried on powered axles, which is what generates adhesion."""
        return self.mass_kg(load_factor) * self.adhesive_mass_fraction

    # ------------------------------------------------------------- geometry

    @property
    def frontal_area_m2(self) -> float:
        """Body width x height above rail.

        An upper bound: the true projected area is slightly smaller because of
        body taper and roof curvature. Overestimating frontal area makes the
        model marginally slower at high speed, which is the conservative
        direction for a project expecting a fast bias.
        """
        return self.body_width_m * self.body_height_m

    @property
    def train_length_m(self) -> float:
        return self.n_cars * self.car_length_m

    # ---------------------------------------------------------- performance

    def tractive_effort_limit(self, resistance_at_rest_n: float,
                              load_factor: float = 0.0) -> float:
        """Maximum tractive effort in the constant-effort region, in newtons.

        Derived from the published starting acceleration rather than assumed.
        The published figure (3.0 km/h/s for the E235) is a NET acceleration --
        it is what the train actually achieves, so running resistance at rest
        is already subtracted. Recovering the gross effort therefore means
        adding it back:

            F_max = m_eff * a_start + R(0)

        The load condition for the published acceleration figure is not stated;
        `load_factor` is the assumption, and it should be swept in sensitivity
        analysis rather than trusted.
        """
        return self.effective_mass_kg(load_factor) * self.start_accel_ms2 + resistance_at_rest_n

    def base_speed_ms(self, resistance_at_rest_n: float,
                      load_factor: float = 0.0) -> float:
        """Speed at which constant-effort gives way to constant-power.

        Below this the traction system is current/adhesion limited and effort is
        flat; above it the system is power limited and F = P/v.

        Caveat worth stating plainly: because `total_power_w` is the ONE-HOUR
        rating and not the short-term overload capability of the inverters, this
        base speed comes out lower than the real machine's. The model is
        therefore pessimistic through the mid-speed region. That works AGAINST
        the expected result (a simulation faster than the timetable), so any
        measured fast bias is a lower bound, not an inflated one.
        """
        return self.total_power_w / self.tractive_effort_limit(resistance_at_rest_n, load_factor)

    def adhesion_limit_n(self, v_ms: float, load_factor: float = 0.0) -> float:
        """Ceiling on tractive effort set by wheel-rail friction, in newtons."""
        return adhesion_coefficient(v_ms) * self.adhesive_mass_kg(load_factor) * G

    # -------------------------------------------------------------- loading

    @classmethod
    def from_json(cls, path: Path | str, key: str = "E235-0") -> "TrainSpec":
        """Build a spec from the provenance-tagged JSON data file."""
        with open(path, encoding="utf-8") as fh:
            blob = json.load(fh)
        d = blob[key]

        def val(*path_parts):
            node = d
            for part in path_parts:
                node = node[part]
            return node["value"] if isinstance(node, dict) and "value" in node else node

        return cls(
            name=d["designation"],
            n_cars=d["formation"]["n_cars"],
            n_motor_cars=d["formation"]["n_motor_cars"],
            tare_mass_kg=val("tare_mass_t") * 1000.0,
            capacity=int(val("capacity_passengers")),
            passenger_mass_kg=val("passenger_mass_kg"),
            total_power_w=val("total_power_kw") * 1000.0,
            max_speed_ms=ms(val("max_operating_speed_kmh")),
            start_accel_ms2=ms2(val("starting_acceleration_kmh_s")),
            service_decel_ms2=ms2(val("service_deceleration_kmh_s")),
            lambda_rot=val("rotational_inertia_lambda"),
            adhesive_mass_fraction=val("adhesive_mass_fraction"),
            body_width_m=val("body_width_mm") / 1000.0,
            body_height_m=val("body_height_mm") / 1000.0,
            car_length_m=val("car_length_intermediate_mm") / 1000.0,
        )


def adhesion_coefficient(v_ms: float) -> float:
    """Wheel-rail adhesion coefficient as a function of speed.

    Curtius-Kniffler, the standard empirical relation:

        mu(v) = 0.161 + 7.5 / (v_kmh + 44)

    Gives about 0.33 at rest falling to about 0.22 at 90 km/h. This is a
    dry-rail figure. Wet or contaminated rail is substantially lower, and
    modelling only dry rail is a stated assumption of this project.
    """
    v_kmh = max(v_ms, 0.0) * 3.6
    return 0.161 + 7.5 / (v_kmh + 44.0)
