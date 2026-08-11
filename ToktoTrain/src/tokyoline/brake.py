"""Jerk-limited braking.

A real brake application cannot step instantly to full deceleration -- that
would be violently uncomfortable for standing passengers, and on a line where
most riders stand, comfort is a hard operational constraint rather than a nicety.
So a stop is three phases:

    ramp in at the jerk limit  ->  hold at maximum deceleration  ->  ramp out

The ramp-out matters as much as the ramp-in: deceleration must reach zero at the
same moment speed does, otherwise the train stops with a lurch.

Consequences that make this worth modelling from the start rather than bolting
on later:

* Stopping takes longer than the idealised instant application by exactly
  ``a_max / j`` seconds in the trapezoidal case -- independent of initial speed.
  For the E235 at 4.2 km/h/s with a 0.75 m/s^3 jerk limit that is about 1.6 s
  per stop, so roughly 47 s over a 30-station circuit against a scheduled
  3948 s. About 1.2 percent: real, measurable, and smaller than the recovery
  margin it must not be confused with.
* It changes the structure of the solver. The brake application point is no
  longer a simple v^2/2a lookup, so it has to be found by intersecting a
  backward-integrated brake curve with the forward run curve.

Sign convention throughout: `a_max` and `jerk` are positive magnitudes.
Deceleration is applied as a negative acceleration internally.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Comfort jerk limit for standing passengers, m/s^3.
#: Literature puts the acceptable range at roughly 0.5-1.0; 0.75 is the midpoint.
#: MODELLED, not published for this line or class.
DEFAULT_JERK_LIMIT = 0.75


@dataclass(frozen=True)
class BrakeProfile:
    """A jerk-limited deceleration profile to a stand.

    Parameters
    ----------
    a_max : maximum deceleration magnitude, m/s^2.
    jerk : maximum rate of change of deceleration, m/s^3.

    The published deceleration figure for Japanese stock (4.2 km/h/s for the
    E235) is a NET rate -- what the train actually achieves. Running resistance
    is therefore not added on top of it during braking, unlike during coasting
    where resistance is the only retarding force. Treating the published figure
    as net is the standard reading and is the conservative one here: it does not
    credit the model with free deceleration.
    """

    a_max: float
    jerk: float = DEFAULT_JERK_LIMIT

    def __post_init__(self) -> None:
        if self.a_max <= 0.0:
            raise ValueError("a_max must be a positive magnitude")
        if self.jerk <= 0.0:
            raise ValueError("jerk must be a positive magnitude")

    @property
    def speed_threshold_for_full_braking(self) -> float:
        """Initial speed below which the profile is triangular, not trapezoidal.

        Below ``a_max^2 / j`` there is not enough speed to reach full
        deceleration before the ramp-out must begin, so the hold phase vanishes.
        """
        return self.a_max ** 2 / self.jerk

    def is_trapezoidal(self, v0: float) -> bool:
        return v0 >= self.speed_threshold_for_full_braking

    def peak_deceleration(self, v0: float) -> float:
        """Actual peak deceleration reached when stopping from `v0`.

        Equals `a_max` for a trapezoidal stop; less for a triangular one.
        """
        if self.is_trapezoidal(v0):
            return self.a_max
        return math.sqrt(max(v0, 0.0) * self.jerk)

    # ------------------------------------------------------- closed solution

    def phase_durations(self, v0: float) -> tuple[float, float, float]:
        """Durations of (ramp-in, hold, ramp-out) when stopping from `v0`."""
        v0 = max(v0, 0.0)
        if v0 == 0.0:
            return (0.0, 0.0, 0.0)

        a_pk = self.peak_deceleration(v0)
        t_ramp = a_pk / self.jerk

        if self.is_trapezoidal(v0):
            # Speed shed by the two ramps together is a_max^2 / j.
            t_hold = (v0 - self.a_max ** 2 / self.jerk) / self.a_max
        else:
            t_hold = 0.0

        return (t_ramp, t_hold, t_ramp)

    def time_to_stop(self, v0: float) -> float:
        """Total time to come to a stand from `v0`, in seconds.

        For a trapezoidal stop this reduces to ``v0 / a_max + a_max / j`` --
        the idealised time plus a fixed jerk penalty.
        """
        t1, t2, t3 = self.phase_durations(v0)
        return t1 + t2 + t3

    def distance_to_stop(self, v0: float) -> float:
        """Distance required to come to a stand from `v0`, in metres.

        Closed form, integrated phase by phase. This is the function the segment
        solver inverts to find the brake application point, and computing it
        analytically rather than by forward trial-and-error is what stops the
        solver overshooting the platform.
        """
        v0 = max(v0, 0.0)
        if v0 == 0.0:
            return 0.0

        j = self.jerk
        t1, t2, t3 = self.phase_durations(v0)
        a_pk = self.peak_deceleration(v0)

        # Phase 1: a ramps 0 -> -a_pk.  v = v0 - 0.5*j*t^2
        d1 = v0 * t1 - j * t1 ** 3 / 6.0
        v1 = v0 - 0.5 * j * t1 ** 2

        # Phase 2: constant -a_pk.
        d2 = v1 * t2 - 0.5 * a_pk * t2 ** 2
        v2 = v1 - a_pk * t2

        # Phase 3: a ramps -a_pk -> 0, ending exactly at v = 0.
        d3 = v2 * t3 - 0.5 * a_pk * t3 ** 2 + j * t3 ** 3 / 6.0

        return d1 + d2 + d3

    def idealised_distance_to_stop(self, v0: float) -> float:
        """Distance under an instant full-brake application: v0^2 / (2*a_max).

        Only useful for quantifying what the jerk limit costs. Never used in the
        simulation itself.
        """
        return max(v0, 0.0) ** 2 / (2.0 * self.a_max)

    def jerk_penalty_distance(self, v0: float) -> float:
        """Extra stopping distance attributable to the jerk limit, in metres."""
        return self.distance_to_stop(v0) - self.idealised_distance_to_stop(v0)

    # ------------------------------------------------------------ trajectory

    def trajectory(self, v0: float, dt: float = 0.01) -> dict[str, np.ndarray]:
        """Sampled deceleration trajectory from `v0` to rest.

        Integrated backwards in spirit but evaluated forwards from the closed
        form, so it carries no integration error of its own. Returned distances
        are measured from the brake application point.

        Returns arrays keyed 't', 'v', 'x', 'a'.
        """
        t_total = self.time_to_stop(v0)
        n = max(int(math.ceil(t_total / dt)) + 1, 2)
        t = np.linspace(0.0, t_total, n)

        j = self.jerk
        t1, t2, _t3 = self.phase_durations(v0)
        a_pk = self.peak_deceleration(v0)

        a = np.empty_like(t)
        v = np.empty_like(t)
        x = np.empty_like(t)

        v1 = v0 - 0.5 * j * t1 ** 2
        d1 = v0 * t1 - j * t1 ** 3 / 6.0
        v2 = v1 - a_pk * t2
        d2 = v1 * t2 - 0.5 * a_pk * t2 ** 2

        in1 = t <= t1
        in2 = (t > t1) & (t <= t1 + t2)
        in3 = t > t1 + t2

        # Ramp in.
        tau = t[in1]
        a[in1] = -j * tau
        v[in1] = v0 - 0.5 * j * tau ** 2
        x[in1] = v0 * tau - j * tau ** 3 / 6.0

        # Hold.
        tau = t[in2] - t1
        a[in2] = -a_pk
        v[in2] = v1 - a_pk * tau
        x[in2] = d1 + v1 * tau - 0.5 * a_pk * tau ** 2

        # Ramp out.
        tau = t[in3] - t1 - t2
        a[in3] = -a_pk + j * tau
        v[in3] = v2 - a_pk * tau + 0.5 * j * tau ** 2
        x[in3] = d1 + d2 + v2 * tau - 0.5 * a_pk * tau ** 2 + j * tau ** 3 / 6.0

        # Clamp the final sample: floating point can leave v a hair below zero.
        v[-1] = 0.0
        a[-1] = 0.0

        return {"t": t, "v": np.maximum(v, 0.0), "x": x, "a": a}
