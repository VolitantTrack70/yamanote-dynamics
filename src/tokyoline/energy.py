"""Phase 5: traction energy accounting.

Traction energy is the integral of tractive effort against speed. The useful
part is not the total but the breakdown: where does it end up?

Over one segment the train starts and ends at rest, so the kinetic energy term
vanishes from the balance and everything the motors put in must leave through
one of two doors:

    E_traction  =  E_resistance  +  E_brake_system

That identity is not an assumption -- it follows from the work-energy theorem
given the start and end conditions -- so computing both sides independently and
checking they agree is a genuine test of the integrator, not a tautology. The
residual is reported with every breakdown, and it should be a rounding error.

A subtlety this module has to be careful about. The published deceleration figure
for Japanese stock is a NET rate: it is what the train actually achieves, with
running resistance already helping. So during braking the brake system supplies
less force than `m_eff * a` -- it supplies `m_eff * a - R(v)`. Attributing the
whole of `m_eff * a` to the brakes would overstate the energy available for
regeneration, and therefore overstate the recovery. The split is done properly
here.

Regeneration parameters are MODELLED. JR East does not publish a recovery
fraction for the E235 on the Yamanote.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .segment import Phase, SegmentResult
from .traction import TractionModel

#: Speed below which regenerative braking is assumed unavailable and friction
#: brakes take over, km/h. MODELLED. Real changeover is a control-system detail
#: that varies with the brake blending strategy and is not published.
REGEN_CUTOFF_KMH = 5.0

#: Motor-plus-inverter conversion efficiency, wheel to pantograph. MODELLED.
REGEN_EFFICIENCY = 0.85

#: Fraction of returned energy actually absorbed rather than burned off. On a
#: line with roughly 2.5 minute headway there is almost always another train
#: drawing current nearby, so receptivity is high -- but this is MODELLED, and
#: it is the least defensible number in this module.
REGEN_RECEPTIVITY = 0.70

#: Auxiliary load: HVAC, lighting, compressors, control, for an 11-car set, kW.
#: MODELLED. Material -- roughly a tenth of traction energy over a circuit -- so
#: omitting it would understate net consumption noticeably.
AUXILIARY_KW = 150.0

JOULES_PER_KWH = 3.6e6


@dataclass
class EnergyBreakdown:
    """Where the energy went, over one segment or a whole circuit, in joules."""

    traction_j: float
    """Work done by the motors at the wheel."""

    resistance_j: float
    """Dissipated against running resistance, over the entire run."""

    brake_system_j: float
    """Absorbed by the brake system: friction plus regenerative."""

    regenerated_j: float
    """Returned to the supply and actually used, after efficiency and receptivity."""

    auxiliary_j: float
    """Hotel load. Independent of how the train is driven."""

    kinetic_peak_j: float
    """Kinetic energy at the brake application point -- the peak of the segment."""

    balance_residual_j: float
    """E_traction - (E_resistance + E_brake_system). Should be ~0."""

    duration_s: float = 0.0
    distance_m: float = 0.0

    @property
    def net_j(self) -> float:
        """Energy drawn from the supply: traction plus auxiliaries, less recovery."""
        return self.traction_j + self.auxiliary_j - self.regenerated_j

    @property
    def gross_j(self) -> float:
        return self.traction_j + self.auxiliary_j

    @property
    def regen_saving_fraction(self) -> float:
        return self.regenerated_j / self.gross_j if self.gross_j else 0.0

    @property
    def balance_error_fraction(self) -> float:
        return abs(self.balance_residual_j) / self.traction_j if self.traction_j else 0.0

    def kwh(self, attr: str) -> float:
        return getattr(self, attr) / JOULES_PER_KWH

    def kwh_per_km(self) -> float:
        """Net traction energy per route km. The headline efficiency figure."""
        return (self.net_j / JOULES_PER_KWH) / (self.distance_m / 1000.0)

    def kwh_per_car_km(self, n_cars: int) -> float:
        """Per car-km, which is how operators usually publish consumption.

        Comparable across trains of different lengths, and the form in which
        published metro figures (typically 2-4 kWh per car-km) are quoted.
        """
        return self.kwh_per_km() / n_cars

    def wh_per_passenger_km(self, passengers: float) -> float:
        """Net energy per passenger-km at a given loading.

        Strongly dependent on occupancy, so it must be quoted with the loading
        assumed. At crush loading a metro is among the most efficient ways to
        move a person; at 3am with six passengers aboard it is not.
        """
        if passengers <= 0:
            return float("inf")
        return (self.net_j / JOULES_PER_KWH * 1000.0) / (self.distance_m / 1000.0) / passengers

    def __add__(self, other: "EnergyBreakdown") -> "EnergyBreakdown":
        return EnergyBreakdown(
            traction_j=self.traction_j + other.traction_j,
            resistance_j=self.resistance_j + other.resistance_j,
            brake_system_j=self.brake_system_j + other.brake_system_j,
            regenerated_j=self.regenerated_j + other.regenerated_j,
            auxiliary_j=self.auxiliary_j + other.auxiliary_j,
            kinetic_peak_j=max(self.kinetic_peak_j, other.kinetic_peak_j),
            balance_residual_j=self.balance_residual_j + other.balance_residual_j,
            duration_s=self.duration_s + other.duration_s,
            distance_m=self.distance_m + other.distance_m,
        )


@dataclass
class EnergyProfile:
    """Cumulative energy against distance, for plotting over the circuit."""

    distance_m: np.ndarray
    traction_j: np.ndarray
    resistance_j: np.ndarray
    regenerated_j: np.ndarray
    net_j: np.ndarray


def _cumtrapz(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Cumulative trapezoidal integral, same length as the inputs, starting at 0."""
    if len(x) < 2:
        return np.zeros_like(x)
    steps = np.diff(x) * 0.5 * (y[1:] + y[:-1])
    return np.concatenate([[0.0], np.cumsum(steps)])


def segment_energy(
    result: SegmentResult,
    traction: TractionModel,
    *,
    regen_efficiency: float = REGEN_EFFICIENCY,
    regen_receptivity: float = REGEN_RECEPTIVITY,
    regen_cutoff_kmh: float = REGEN_CUTOFF_KMH,
    auxiliary_kw: float = AUXILIARY_KW,
    include_dwell_s: float = 0.0,
) -> tuple[EnergyBreakdown, EnergyProfile]:
    """Account for the energy used over one interstation run.

    `include_dwell_s` adds auxiliary load for a stationary period after the run,
    which matters over a circuit: the hotel load runs during dwell too, and on a
    line where roughly a quarter of the schedule is spent standing at platforms
    that is not a negligible correction.
    """
    t = result.t
    v = result.v
    m_eff = traction.spec.effective_mass_kg(traction.load_factor)

    # Instantaneous powers, watts.
    p_traction = result.tractive_force_n * v
    resistance_n = np.array([traction.davis(vi) for vi in v])
    p_resistance = resistance_n * v

    is_brake = np.array([p is Phase.BRAKE for p in result.phase])

    # Brake system force is the NET commanded deceleration less the help already
    # being given by running resistance. Clamped at zero: during the jerk ramps
    # the commanded rate is small enough that resistance alone can exceed it, and
    # a brake cannot absorb negative energy.
    decel_force = m_eff * np.abs(result.a) * is_brake
    p_brake_system = np.maximum(decel_force - resistance_n, 0.0) * v * is_brake

    # Regeneration is only available above the changeover speed.
    regen_available = is_brake & (v * 3.6 >= regen_cutoff_kmh)
    p_regen = p_brake_system * regen_available * regen_efficiency * regen_receptivity

    cum_traction = _cumtrapz(p_traction, t)
    cum_resistance = _cumtrapz(p_resistance, t)
    cum_brake = _cumtrapz(p_brake_system, t)
    cum_regen = _cumtrapz(p_regen, t)

    aux_w = auxiliary_kw * 1000.0
    cum_aux = aux_w * t
    duration = float(t[-1])

    e_traction = float(cum_traction[-1])
    e_resistance = float(cum_resistance[-1])
    e_brake = float(cum_brake[-1])
    e_regen = float(cum_regen[-1])
    e_aux = aux_w * (duration + include_dwell_s)

    kinetic_peak = 0.5 * m_eff * float(v.max()) ** 2

    breakdown = EnergyBreakdown(
        traction_j=e_traction,
        resistance_j=e_resistance,
        brake_system_j=e_brake,
        regenerated_j=e_regen,
        auxiliary_j=e_aux,
        kinetic_peak_j=kinetic_peak,
        balance_residual_j=e_traction - (e_resistance + e_brake),
        duration_s=duration + include_dwell_s,
        distance_m=result.distance_m,
    )

    profile = EnergyProfile(
        distance_m=result.x,
        traction_j=cum_traction,
        resistance_j=cum_resistance,
        regenerated_j=cum_regen,
        net_j=cum_traction + aux_w * t - cum_regen,
    )
    return breakdown, profile


def circuit_energy(
    results: list[SegmentResult],
    traction: TractionModel,
    *,
    dwell_s: float = 0.0,
    **kwargs,
) -> tuple[EnergyBreakdown, EnergyProfile]:
    """Accumulate energy over every segment of a circuit.

    Returns the summed breakdown and a single cumulative profile whose distance
    axis runs continuously around the loop.
    """
    total: EnergyBreakdown | None = None
    xs: list[np.ndarray] = []
    tr: list[np.ndarray] = []
    rs: list[np.ndarray] = []
    rg: list[np.ndarray] = []
    nt: list[np.ndarray] = []

    x_offset = 0.0
    tr_off = rs_off = rg_off = nt_off = 0.0

    for r in results:
        b, p = segment_energy(r, traction, include_dwell_s=dwell_s, **kwargs)
        total = b if total is None else total + b

        xs.append(p.distance_m + x_offset)
        tr.append(p.traction_j + tr_off)
        rs.append(p.resistance_j + rs_off)
        rg.append(p.regenerated_j + rg_off)
        nt.append(p.net_j + nt_off)

        x_offset += r.distance_m
        tr_off = tr[-1][-1]
        rs_off = rs[-1][-1]
        rg_off = rg[-1][-1]
        # Carry the dwell auxiliary load across the join so the net curve does
        # not silently lose it at every station.
        nt_off = nt[-1][-1] + kwargs.get("auxiliary_kw", AUXILIARY_KW) * 1000.0 * dwell_s

    assert total is not None
    profile = EnergyProfile(
        distance_m=np.concatenate(xs),
        traction_j=np.concatenate(tr),
        resistance_j=np.concatenate(rs),
        regenerated_j=np.concatenate(rg),
        net_j=np.concatenate(nt),
    )
    return total, profile
