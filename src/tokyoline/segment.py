"""Single-segment run simulation: fixed-step RK4 with control-phase switching.

The control sequence is accelerate -> cruise -> (optionally coast) -> brake.

Why fixed-step rather than an adaptive solver: the interesting difficulty here
is not stiffness, it is knowing exactly when to switch control phase. A fixed
step makes phase transitions easy to reason about and easy to debug, and the
step-size convergence study is then a one-line sweep. `scipy.integrate.solve_ivp`
would handle the smooth parts more efficiently and the switching less clearly.

The brake application point is found by intersecting the forward run curve with a
brake curve integrated backwards from rest at the platform. Concretely: the
trigger is ``x + distance_to_stop(v) >= L``, and the crossing is located by
bisection inside the step where it first becomes true, then the exact closed-form
brake trajectory is attached. Solving it this way is what keeps the train from
overshooting the platform -- a purely forward solver that brakes only once it has
already gone too far will always overshoot, and the overshoot grows with step
size.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .brake import BrakeProfile
from .traction import TractionModel


class Phase(str, Enum):
    ACCEL = "accel"
    CRUISE = "cruise"
    COAST = "coast"
    BRAKE = "brake"


@dataclass
class SegmentResult:
    """Trajectory and summary for one interstation run."""

    run_time_s: float
    distance_m: float
    t: np.ndarray
    x: np.ndarray
    v: np.ndarray
    a: np.ndarray
    phase: list[Phase]
    tractive_force_n: np.ndarray

    max_speed_ms: float = 0.0
    position_error_m: float = 0.0
    """Distance by which the train missed the platform. Should be ~1e-9."""

    phase_times_s: dict[str, float] = field(default_factory=dict)
    phase_distances_m: dict[str, float] = field(default_factory=dict)

    @property
    def max_speed_kmh(self) -> float:
        return self.max_speed_ms * 3.6

    @property
    def reached_line_speed(self) -> bool:
        return self.phase_times_s.get(Phase.CRUISE.value, 0.0) > 0.0

    def summary(self) -> str:
        parts = ", ".join(
            f"{k}={v:.1f}s" for k, v in self.phase_times_s.items() if v > 0.0
        )
        return (
            f"{self.distance_m:.0f} m in {self.run_time_s:.1f} s "
            f"(peak {self.max_speed_kmh:.1f} km/h; {parts})"
        )


def _applied_force(traction: TractionModel, v: float, phase: "Phase",
                   kinematics: dict) -> float:
    """Tractive force actually applied in a given control phase, in newtons.

    Recorded so that energy accounting integrates the real force rather than the
    force available. The distinction only bites during cruise, where the driver
    throttles back to balance resistance -- and where `effort_n` would report
    zero, because cruise on this line happens exactly at maximum speed.
    """
    if phase is Phase.ACCEL:
        return traction.effort_n(v)
    if phase is Phase.CRUISE:
        return traction.holding_force_n(v, **kinematics)
    return 0.0  # coasting and braking apply no tractive effort


def _rk4_step(x: float, v: float, dt: float, accel_fn) -> tuple[float, float]:
    """One fixed-step RK4 update of (position, speed).

    `accel_fn(v)` returns acceleration in m/s^2. Acceleration has no explicit
    position dependence on flat track with a constant speed limit, which is why
    it takes speed alone.
    """
    k1v = accel_fn(v)
    k1x = v

    k2v = accel_fn(v + 0.5 * dt * k1v)
    k2x = v + 0.5 * dt * k1v

    k3v = accel_fn(v + 0.5 * dt * k2v)
    k3x = v + 0.5 * dt * k2v

    k4v = accel_fn(v + dt * k3v)
    k4x = v + dt * k3v

    v_new = v + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
    x_new = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
    return x_new, max(v_new, 0.0)


def simulate_segment(
    traction: TractionModel,
    brake: BrakeProfile,
    distance_m: float,
    *,
    speed_limit_ms: float | None = None,
    dt: float = 0.05,
    coast_start_m: float | None = None,
    grade_permille: float = 0.0,
    curve_radius_m: float | None = None,
    max_steps: int = 2_000_000,
) -> SegmentResult:
    """Simulate one station-to-station run, starting and ending at rest.

    Parameters
    ----------
    distance_m : interstation distance.
    speed_limit_ms : line speed limit. Defaults to the train's own maximum.
        Per-segment restrictions from curves, turnouts and signalling are not
        public for the Yamanote and are not modelled -- a stated limitation.
    dt : fixed integration step, seconds. 0.01-0.1 is the sensible range.
    coast_start_m : position at which to cut power and coast. `None` means no
        coasting, which gives the minimum run time. This is the single free
        parameter the optimal-driving solver bisects on.

    Returns the full trajectory plus a run time. The run time is a MINIMUM under
    the stated assumptions: full available tractive effort throughout, braking at
    the maximum service rate, no signal restrictions, no driver conservatism and
    no recovery margin. It is meant to be compared against a real timetable, not
    to reproduce one.
    """
    if distance_m <= 0.0:
        raise ValueError("distance_m must be positive")

    v_limit = speed_limit_ms if speed_limit_ms is not None else traction.spec.max_speed_ms
    v_limit = min(v_limit, traction.spec.max_speed_ms)

    kinematics = dict(grade_permille=grade_permille, curve_radius_m=curve_radius_m)

    def accel_powering(v: float) -> float:
        return traction.acceleration_ms2(v, powering=True, **kinematics)

    def accel_coasting(v: float) -> float:
        return traction.acceleration_ms2(v, powering=False, **kinematics)

    def brake_trigger(x: float, v: float) -> float:
        """Positive once the train must brake to stop at the platform."""
        return x + brake.distance_to_stop(v) - distance_m

    if brake_trigger(0.0, 0.0) > 0.0:
        raise ValueError("segment shorter than the minimum stopping distance from rest")

    ts: list[float] = [0.0]
    xs: list[float] = [0.0]
    vs: list[float] = [0.0]
    accs: list[float] = [accel_powering(0.0)]
    forces: list[float] = [traction.effort_n(0.0)]
    phases: list[Phase] = [Phase.ACCEL]

    t = 0.0
    x = 0.0
    v = 0.0
    steps = 0

    # ---- powered / cruising / coasting run, until the brake curve is met ----
    while steps < max_steps:
        coasting = coast_start_m is not None and x >= coast_start_m

        if coasting:
            phase = Phase.COAST
            accel_fn = accel_coasting
        elif v >= v_limit - 1e-9:
            phase = Phase.CRUISE
            accel_fn = None  # speed held constant
        else:
            phase = Phase.ACCEL
            accel_fn = accel_powering

        if accel_fn is None:
            # Cruise: driver holds the limit exactly, so tractive effort matches
            # resistance and acceleration is zero by construction.
            x_new = x + v * dt
            v_new = v
        else:
            x_new, v_new = _rk4_step(x, v, dt, accel_fn)
            if not coasting and v_new > v_limit:
                v_new = v_limit

        # Did we cross the brake curve inside this step? Locate it precisely --
        # this is the forward/backward intersection the whole solver hinges on.
        if brake_trigger(x_new, v_new) >= 0.0:
            lo, hi = 0.0, dt
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if accel_fn is None:
                    xm, vm = x + v * mid, v
                else:
                    xm, vm = _rk4_step(x, v, mid, accel_fn)
                    if not coasting and vm > v_limit:
                        vm = v_limit
                if brake_trigger(xm, vm) >= 0.0:
                    hi = mid
                else:
                    lo = mid
            if accel_fn is None:
                x, v = x + v * hi, v
            else:
                x, v = _rk4_step(x, v, hi, accel_fn)
                if not coasting and v > v_limit:
                    v = v_limit
            t += hi

            ts.append(t)
            xs.append(x)
            vs.append(v)
            accs.append(0.0 if accel_fn is None else accel_fn(v))
            forces.append(_applied_force(traction, v, phase, kinematics))
            phases.append(phase)
            break

        t += dt
        x, v = x_new, v_new
        steps += 1

        ts.append(t)
        xs.append(x)
        vs.append(v)
        accs.append(0.0 if accel_fn is None else accel_fn(v))
        forces.append(_applied_force(traction, v, phase, kinematics))
        phases.append(phase)
    else:
        raise RuntimeError("segment did not converge within max_steps")

    # ---- attach the exact closed-form brake trajectory ----
    traj = brake.trajectory(v, dt=dt)
    # Skip its first sample: it duplicates the state we already recorded.
    for i in range(1, len(traj["t"])):
        ts.append(t + traj["t"][i])
        xs.append(x + traj["x"][i])
        vs.append(traj["v"][i])
        accs.append(traj["a"][i])
        forces.append(0.0)
        phases.append(Phase.BRAKE)

    t_arr = np.asarray(ts)
    x_arr = np.asarray(xs)
    v_arr = np.asarray(vs)

    phase_times: dict[str, float] = {}
    phase_distances: dict[str, float] = {}
    for p in Phase:
        mask = np.array([ph is p for ph in phases])
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            phase_times[p.value] = 0.0
            phase_distances[p.value] = 0.0
            continue
        # Attribute each interval to the phase active at its start.
        dur = 0.0
        dist = 0.0
        for i in idx:
            if i + 1 < len(t_arr):
                dur += t_arr[i + 1] - t_arr[i]
                dist += x_arr[i + 1] - x_arr[i]
        phase_times[p.value] = dur
        phase_distances[p.value] = dist

    return SegmentResult(
        run_time_s=float(t_arr[-1]),
        distance_m=distance_m,
        t=t_arr,
        x=x_arr,
        v=v_arr,
        a=np.asarray(accs),
        phase=phases,
        tractive_force_n=np.asarray(forces),
        max_speed_ms=float(v_arr.max()),
        position_error_m=float(x_arr[-1] - distance_m),
        phase_times_s=phase_times,
        phase_distances_m=phase_distances,
    )
