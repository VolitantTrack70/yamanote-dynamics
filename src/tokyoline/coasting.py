"""Phase 6: optimal coasting and the energy-vs-run-time trade-off.

The known result from optimal control applied to train operation is that the
energy-minimal way to cover a fixed distance in a fixed time is a four-phase
profile:

    maximum acceleration -> cruise -> coast -> maximum braking

This module does not derive that result. It takes it as given and builds the
solver, which is the useful part: with the shape fixed, the whole strategy
collapses to ONE number per segment -- where coasting starts. Run time falls
monotonically as that point moves later, so a bisection finds the coast onset
that hits any achievable target time, and sweeping the target traces out the
Pareto front. A one-dimensional search, not Pontryagin.

**The comparison that makes it meaningful.** Showing that coasting saves energy
against a flat-out run is not interesting, because the flat-out run is faster --
of course it uses more. The honest comparison holds run time equal. A driver who
wants to take longer without coasting has an obvious alternative: cruise slower.
So this module solves both strategies to the same target time and compares:

    coast   : accelerate to line speed, cruise, cut power, drift, brake
    slower  : accelerate to a LOWER cruise speed, hold it, brake

Both arrive at the same moment. The energy difference between them is the real
saving attributable to coasting.

**An interaction worth watching.** Coasting and regeneration are partial
substitutes. Coasting saves energy by not spending it; regeneration saves energy
by recovering what was spent. The more effective the regeneration, the less
coasting adds on top. The scripts report the saving at several receptivities for
exactly this reason.
"""

from __future__ import annotations

from dataclasses import dataclass

from .brake import BrakeProfile
from .energy import JOULES_PER_KWH, segment_energy
from .segment import simulate_segment
from .traction import TractionModel


@dataclass(frozen=True)
class Strategy:
    """One driving strategy evaluated on one segment."""

    label: str
    run_time_s: float
    net_energy_j: float
    traction_energy_j: float
    regenerated_j: float
    coast_start_m: float | None
    cruise_speed_kmh: float
    peak_speed_kmh: float
    coast_distance_m: float = 0.0

    @property
    def net_kwh(self) -> float:
        return self.net_energy_j / JOULES_PER_KWH


def evaluate(
    traction: TractionModel,
    brake: BrakeProfile,
    distance_m: float,
    *,
    speed_limit_ms: float,
    coast_start_m: float | None = None,
    dt: float = 0.1,
    label: str = "",
    **energy_opts,
) -> Strategy:
    """Run one segment under one strategy and account for its energy."""
    r = simulate_segment(traction, brake, distance_m,
                         speed_limit_ms=speed_limit_ms, dt=dt,
                         coast_start_m=coast_start_m)
    b, _ = segment_energy(r, traction, **energy_opts)
    return Strategy(
        label=label,
        run_time_s=r.run_time_s,
        net_energy_j=b.net_j,
        traction_energy_j=b.traction_j,
        regenerated_j=b.regenerated_j,
        coast_start_m=coast_start_m,
        cruise_speed_kmh=speed_limit_ms * 3.6,
        peak_speed_kmh=r.max_speed_kmh,
        coast_distance_m=r.phase_distances_m.get("coast", 0.0),
    )


def _bisect(f, lo: float, hi: float, target: float, *, tol: float,
            max_iter: int = 60, decreasing: bool = True) -> float:
    """Find x in [lo, hi] with f(x) ~ target, for a monotonic f.

    Plain bisection rather than a library root-finder: f involves a full
    segment simulation, is only piecewise smooth (it goes flat once the coast
    point moves past the brake application point), and a bracketing method that
    cannot overshoot is the right tool for that.
    """
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        mid = 0.5 * (lo + hi)
        value = f(mid)
        if (value > target) == decreasing:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def solve_coast_for_time(
    traction: TractionModel,
    brake: BrakeProfile,
    distance_m: float,
    target_time_s: float,
    *,
    speed_limit_ms: float,
    dt: float = 0.1,
    tol_m: float = 1.0,
    **energy_opts,
) -> Strategy | None:
    """Find the coast onset that achieves `target_time_s`.

    Returns None when the target is unreachable -- faster than the flat-out run,
    or slower than coasting from the earliest sensible point. Returning None
    rather than the nearest achievable strategy keeps an infeasible request from
    silently becoming a feasible-looking answer.
    """
    fastest = evaluate(traction, brake, distance_m, speed_limit_ms=speed_limit_ms,
                       coast_start_m=None, dt=dt, label="flat out", **energy_opts)
    if target_time_s <= fastest.run_time_s:
        return None

    lo = 0.05 * distance_m
    slowest = evaluate(traction, brake, distance_m, speed_limit_ms=speed_limit_ms,
                       coast_start_m=lo, dt=dt, **energy_opts)
    if target_time_s > slowest.run_time_s:
        return None

    def run_time(coast_start: float) -> float:
        return simulate_segment(traction, brake, distance_m,
                                speed_limit_ms=speed_limit_ms, dt=dt,
                                coast_start_m=coast_start).run_time_s

    x = _bisect(run_time, lo, distance_m, target_time_s, tol=tol_m, decreasing=True)
    return evaluate(traction, brake, distance_m, speed_limit_ms=speed_limit_ms,
                    coast_start_m=x, dt=dt, label="coast", **energy_opts)


def solve_speed_for_time(
    traction: TractionModel,
    brake: BrakeProfile,
    distance_m: float,
    target_time_s: float,
    *,
    speed_limit_ms: float,
    dt: float = 0.1,
    tol_ms: float = 0.02,
    **energy_opts,
) -> Strategy | None:
    """Find the reduced cruise speed that achieves `target_time_s`.

    The naive alternative to coasting: go slower throughout, still full power to
    the cruise speed and full braking at the end. This is what the coasting
    strategy has to beat to be worth anything.
    """
    fastest = evaluate(traction, brake, distance_m, speed_limit_ms=speed_limit_ms,
                       dt=dt, **energy_opts)
    if target_time_s <= fastest.run_time_s:
        return None

    lo = 0.2 * speed_limit_ms

    def run_time(v_limit: float) -> float:
        return simulate_segment(traction, brake, distance_m,
                                speed_limit_ms=v_limit, dt=dt).run_time_s

    if target_time_s > run_time(lo):
        return None

    # Run time falls as the speed limit rises, so this is a DECREASING function
    # of the search variable -- the same orientation as the coast search.
    v = _bisect(run_time, lo, speed_limit_ms, target_time_s, tol=tol_ms,
                decreasing=True)
    return evaluate(traction, brake, distance_m, speed_limit_ms=v, dt=dt,
                    label="slower cruise", **energy_opts)


@dataclass
class ParetoPoint:
    target_time_s: float
    time_factor: float
    coast: Strategy | None
    slower: Strategy | None

    @property
    def saving_j(self) -> float | None:
        """Energy the coasting strategy saves over cruising slower, same time."""
        if self.coast is None or self.slower is None:
            return None
        return self.slower.net_energy_j - self.coast.net_energy_j

    @property
    def saving_fraction(self) -> float | None:
        if self.saving_j is None or not self.slower.net_energy_j:
            return None
        return self.saving_j / self.slower.net_energy_j


def pareto_front(
    traction: TractionModel,
    brake: BrakeProfile,
    distance_m: float,
    *,
    speed_limit_ms: float,
    time_factors: tuple[float, ...] = (1.0, 1.02, 1.05, 1.10, 1.15, 1.20, 1.30),
    dt: float = 0.1,
    **energy_opts,
) -> list[ParetoPoint]:
    """Trace energy against run time, for both strategies.

    `time_factors` are multiples of the flat-out minimum run time. A factor of
    1.10 means "take 10 percent longer than the fastest possible run".
    """
    fastest = evaluate(traction, brake, distance_m, speed_limit_ms=speed_limit_ms,
                       dt=dt, label="flat out", **energy_opts)

    points: list[ParetoPoint] = []
    for factor in time_factors:
        target = fastest.run_time_s * factor
        if factor <= 1.0:
            points.append(ParetoPoint(target, factor, fastest, fastest))
            continue
        points.append(ParetoPoint(
            target_time_s=target,
            time_factor=factor,
            coast=solve_coast_for_time(traction, brake, distance_m, target,
                                       speed_limit_ms=speed_limit_ms, dt=dt,
                                       **energy_opts),
            slower=solve_speed_for_time(traction, brake, distance_m, target,
                                        speed_limit_ms=speed_limit_ms, dt=dt,
                                        **energy_opts),
        ))
    return points
