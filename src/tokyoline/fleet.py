"""Phase 4: cross-class comparison.

The contrast the comparison exists to make legible: a commuter EMU and a
high-speed set are optimised for opposite things. The E235 accelerates at 3.0
km/h/s and tops out at 90; the E5 accelerates at 1.71 and tops out at 320. Neither
is "better" -- they are solving different problems, and a scatter of acceleration
against maximum speed shows the trade-off in one glance.

Three things this module is careful about:

1. **Bounded quantities stay bounded.** N700S formation mass is published only as
   "under 700 t". Power-to-weight computed from it is therefore a LOWER bound,
   and it is returned tagged as one rather than silently becoming a point value.

2. **Missing quantities stay missing.** Service deceleration is not published for
   the E5 or N700S in the sources consulted. They are excluded from braking
   comparisons rather than filled in with a peer value.

3. **Speed-dependent braking is modelled where it is published.** The E7 publishes
   a deceleration that falls from 2.69 km/h/s at low speed to 1.44 at 275 km/h,
   because regenerative brake effort is power-limited just as tractive effort is.
   Treating that as a constant would understate its stopping distance badly. For a
   90 km/h commuter EMU the constant assumption is harmless; for high-speed stock
   it is not, and that difference is itself a result worth showing.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .brake import DEFAULT_JERK_LIMIT, BrakeProfile
from .data import DATA_DIR
from .units import KMHS_TO_MS2, KMH_TO_MS


@dataclass(frozen=True)
class Bounded:
    """A quantity known only as a one-sided bound.

    Exists so that a bound cannot be mistaken for a measurement further
    downstream. Anything derived from it should carry `is_bound` through.
    """

    value: float
    kind: str  # "upper" | "lower"

    def __float__(self) -> float:
        return self.value

    def __str__(self) -> str:
        sign = "<" if self.kind == "upper" else ">"
        return f"{sign} {self.value:g}"


@dataclass(frozen=True)
class FleetMember:
    key: str
    display_name: str
    category: str
    operator: str

    n_cars: int
    n_motor_cars: int
    formation: str

    tare_mass_t: float | None
    tare_mass_bound_t: float | None
    capacity: int

    total_power_kw: float
    max_operating_speed_kmh: float
    design_max_speed_kmh: float
    starting_acceleration_kmh_s: float
    service_deceleration_kmh_s: float | None
    deceleration_curve: list[tuple[float, float]] | None

    car_length_m: float
    body_width_m: float
    body_height_m: float

    decel_is_ambiguous: bool = False

    # ------------------------------------------------------------- derived

    @property
    def motored_fraction(self) -> float:
        return self.n_motor_cars / self.n_cars

    @property
    def frontal_area_m2(self) -> float:
        return self.body_width_m * self.body_height_m

    @property
    def train_length_m(self) -> float:
        return self.n_cars * self.car_length_m

    @property
    def mass_for_ratios_t(self) -> float:
        """Mass to use in ratios. Falls back to the bound when exact is unknown."""
        if self.tare_mass_t is not None:
            return self.tare_mass_t
        if self.tare_mass_bound_t is not None:
            return self.tare_mass_bound_t
        raise ValueError(f"{self.key}: no mass available")

    @property
    def mass_is_bound(self) -> bool:
        return self.tare_mass_t is None and self.tare_mass_bound_t is not None

    def power_to_weight_kw_per_t(self) -> float | Bounded:
        """Installed power per tonne of tare mass.

        Returns a Bounded when formation mass is only bounded: dividing by an
        upper-bound mass yields a LOWER bound on power-to-weight.
        """
        ratio = self.total_power_kw / self.mass_for_ratios_t
        if self.mass_is_bound:
            return Bounded(ratio, "lower")
        return ratio

    def specific_power_kw_per_passenger(self) -> float | Bounded:
        return self.total_power_kw / self.capacity

    # ------------------------------------------------------------- braking

    def deceleration_at_kmh(self, v_kmh: float) -> float | None:
        """Service deceleration at a given speed, in km/h/s.

        Uses the published speed-dependent curve when there is one, otherwise
        the single published figure, otherwise None.
        """
        if self.deceleration_curve:
            xs = [p[0] for p in self.deceleration_curve]
            ys = [p[1] for p in self.deceleration_curve]
            return float(np.interp(v_kmh, xs, ys))
        return self.service_deceleration_kmh_s

    def stopping_distance_m(self, from_kmh: float | None = None,
                            jerk: float = DEFAULT_JERK_LIMIT,
                            dt: float = 0.01) -> float | None:
        """Distance to stop from a given speed under a shared jerk limit.

        Defaults to stopping from the class's own maximum operating speed, which
        is the comparison the spec asks for -- each train braking from its own
        cruise, under identical comfort constraints.

        Returns None when the class publishes no deceleration figure. Callers
        must handle that rather than receiving a plausible-looking default.
        """
        if self.deceleration_at_kmh(0.0) is None:
            return None

        v0 = (from_kmh if from_kmh is not None else self.max_operating_speed_kmh) * KMH_TO_MS

        if not self.deceleration_curve:
            # Constant rate: the closed form in BrakeProfile is exact.
            a_max = self.service_deceleration_kmh_s * KMHS_TO_MS2
            return BrakeProfile(a_max=a_max, jerk=jerk).distance_to_stop(v0)

        return self._variable_rate_stop(v0, jerk, dt)

    def _variable_rate_stop(self, v0: float, jerk: float, dt: float) -> float:
        """Integrate a stop under a speed-dependent deceleration limit.

        Explicit stepping with a jerk-limited ramp in and out. Not RK4: the
        controlling quantity is a rate limit that changes with speed, and the
        ramp-out condition is a switching event, so a small explicit step is both
        adequate and easier to reason about. Step-size sensitivity is checked in
        the fleet script.
        """
        v = v0
        x = 0.0
        a = 0.0  # current deceleration magnitude, m/s^2

        while v > 1e-6:
            target = self.deceleration_at_kmh(v * 3.6) * KMHS_TO_MS2

            # Begin ramping out when the remaining speed is exactly what the
            # ramp-down itself will shed. Same condition as the constant-rate case.
            if v <= a * a / (2.0 * jerk):
                a = max(a - jerk * dt, 0.0)
            elif a < target:
                a = min(a + jerk * dt, target)
            else:
                a = max(a - jerk * dt, target)

            dv = a * dt
            if dv > v:
                # Final partial step: close out on the remaining speed.
                x += v * v / (2.0 * a) if a > 0 else 0.0
                break

            x += v * dt - 0.5 * a * dt * dt
            v -= dv

        return x

    def stopping_time_s(self, from_kmh: float | None = None,
                        jerk: float = DEFAULT_JERK_LIMIT) -> float | None:
        if self.deceleration_at_kmh(0.0) is None:
            return None
        v0 = (from_kmh if from_kmh is not None else self.max_operating_speed_kmh) * KMH_TO_MS
        if not self.deceleration_curve:
            a_max = self.service_deceleration_kmh_s * KMHS_TO_MS2
            return BrakeProfile(a_max=a_max, jerk=jerk).time_to_stop(v0)
        # Recover time from the same integration.
        dt = 0.01
        v, t, a = v0, 0.0, 0.0
        while v > 1e-6:
            target = self.deceleration_at_kmh(v * 3.6) * KMHS_TO_MS2
            if v <= a * a / (2.0 * jerk):
                a = max(a - jerk * dt, 0.0)
            elif a < target:
                a = min(a + jerk * dt, target)
            else:
                a = max(a - jerk * dt, target)
            dv = a * dt
            t += dt
            if dv > v:
                break
            v -= dv
        return t


def to_train_spec(m: FleetMember, *, passenger_mass_kg: float = 60.0,
                  lambda_rot: float = 0.08):
    """Convert a fleet entry into a simulatable :class:`TrainSpec`.

    Raises for classes that cannot be run. That is deliberate: it is better for
    an unusable class to fail loudly than to be silently simulated with an
    invented deceleration figure.
    """
    from .stock import TrainSpec
    from .units import ms, ms2

    if m.service_deceleration_kmh_s is None:
        raise ValueError(
            f"{m.key}: no published service deceleration, cannot be simulated"
        )
    if m.tare_mass_t is None:
        raise ValueError(f"{m.key}: formation mass is only bounded, cannot be simulated")

    return TrainSpec(
        name=m.display_name,
        n_cars=m.n_cars,
        n_motor_cars=m.n_motor_cars,
        tare_mass_kg=m.tare_mass_t * 1000.0,
        capacity=m.capacity,
        passenger_mass_kg=passenger_mass_kg,
        total_power_w=m.total_power_kw * 1000.0,
        max_speed_ms=ms(m.max_operating_speed_kmh),
        start_accel_ms2=ms2(m.starting_acceleration_kmh_s),
        service_decel_ms2=ms2(m.service_deceleration_kmh_s),
        lambda_rot=lambda_rot,
        adhesive_mass_fraction=m.n_motor_cars / m.n_cars,
        body_width_m=m.body_width_m,
        body_height_m=m.body_height_m,
        car_length_m=m.car_length_m,
    )


def simulatable(m: FleetMember) -> tuple[bool, str]:
    """Whether a class can be run on this line's model, and why not if it cannot.

    High-speed stock is excluded on purpose, for two independent reasons:

    * The Davis estimator was built for a blunt-nosed commuter EMU and does not
      transfer to a streamlined 400 m trainset. Running one anyway would produce
      confident-looking numbers from a resistance model known to be wrong.
    * Braking rate for high-speed stock is strongly speed-dependent (the E7
      publishes 2.69 km/h/s at low speed falling to 1.44 at line speed), which
      the segment solver's constant-rate brake profile does not represent.

    Neither is a hard technical barrier -- both are fixable -- but until they
    are fixed the honest answer is to refuse rather than to render.
    """
    if m.category != "commuter EMU":
        return False, ("high-speed stock: the commuter Davis estimator does not "
                       "transfer, and its braking rate is speed-dependent")
    if m.service_deceleration_kmh_s is None:
        return False, "no published service deceleration"
    if m.tare_mass_t is None:
        return False, "formation mass is only bounded"
    return True, ""


def load_fleet(path: Path | str | None = None) -> dict[str, FleetMember]:
    """Load the comparison fleet, preserving bounds and gaps."""
    path = Path(path) if path else DATA_DIR / "fleet.json"
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)

    def val(node, key, default=None):
        n = node.get(key)
        if n is None:
            return default
        return n.get("value", default) if isinstance(n, dict) else n

    out: dict[str, FleetMember] = {}
    for key, d in blob["classes"].items():
        mass_node = d["tare_mass_t"]
        decel_node = d["service_deceleration_kmh_s"]
        curve_node = d.get("deceleration_curve_kmh_s")

        out[key] = FleetMember(
            key=key,
            display_name=d["display_name"],
            category=d["category"],
            operator=d["operator"],
            n_cars=d["formation"]["n_cars"],
            n_motor_cars=d["formation"]["n_motor_cars"],
            formation=d["formation"]["value"],
            tare_mass_t=mass_node.get("value"),
            tare_mass_bound_t=mass_node.get("upper_bound"),
            capacity=int(val(d, "capacity")),
            total_power_kw=float(val(d, "total_power_kw")),
            max_operating_speed_kmh=float(val(d, "max_operating_speed_kmh")),
            design_max_speed_kmh=float(val(d, "design_max_speed_kmh")),
            starting_acceleration_kmh_s=float(val(d, "starting_acceleration_kmh_s")),
            service_deceleration_kmh_s=decel_node.get("value"),
            deceleration_curve=(
                [tuple(p) for p in curve_node["points"]] if curve_node else None
            ),
            car_length_m=float(val(d, "car_length_m")),
            body_width_m=float(val(d, "body_width_m")),
            body_height_m=float(val(d, "body_height_m")),
            decel_is_ambiguous=decel_node.get("status") == "published_ambiguous",
        )
    return out
