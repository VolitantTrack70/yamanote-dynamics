"""Tractive effort as a function of speed.

Three constraints act simultaneously and the delivered effort is the minimum of
all three:

1. Constant-effort region -- below base speed, effort is limited by motor
   current and is flat at F_max.
2. Constant-power region -- above base speed the traction system is power
   limited, so F = P/v, a hyperbola.
3. Adhesion ceiling -- independent of both, F cannot exceed
   mu(v) * m_adhesive * g.

Plotting all three together is one of the more informative single graphs in the
project: for a commuter EMU with 6 of 11 cars powered the adhesion ceiling sits
well clear and never binds, whereas for locomotive-hauled stock it is the
binding constraint at starting.
"""

from __future__ import annotations

from dataclasses import dataclass

from .resistance import DavisCoefficients
from .stock import TrainSpec


@dataclass(frozen=True)
class TractionModel:
    """Tractive effort curve for one train at one load condition."""

    spec: TrainSpec
    davis: DavisCoefficients
    load_factor: float = 1.0
    power_factor: float = 1.0
    """Multiplier on the published one-hour power rating.

    Left at 1.0 by default and it should stay there for the headline result.
    Real traction inverters deliver above the one-hour rating for short
    periods, so 1.0 understates mid-speed performance and makes the simulation
    slower than the real machine. That biases the model AWAY from the expected
    finding, so leaving it at 1.0 keeps any measured fast bias a lower bound.
    Raising it is a sensitivity experiment, not a fit.
    """

    @property
    def resistance_at_rest_n(self) -> float:
        return self.davis(0.0)

    @property
    def max_effort_n(self) -> float:
        """Flat effort in the constant-effort region, in newtons."""
        return self.spec.tractive_effort_limit(self.resistance_at_rest_n, self.load_factor)

    @property
    def power_w(self) -> float:
        return self.spec.total_power_w * self.power_factor

    @property
    def base_speed_ms(self) -> float:
        """Corner between the constant-effort and constant-power regions."""
        return self.power_w / self.max_effort_n

    def capability_n(self, v_ms: float) -> float:
        """Greatest tractive effort the machine can produce at this speed.

        The minimum of the current, power and adhesion limits. Unlike
        :meth:`effort_n` this does NOT cut off at maximum speed, because that
        cutoff is a control decision -- do not accelerate past the limit -- and
        not a statement that the traction system produces no force there. Cruise
        at line speed needs real force to balance resistance.
        """
        v = max(v_ms, 0.0)
        current_limited = self.max_effort_n
        if v <= self.base_speed_ms:
            power_limited = current_limited
        else:
            power_limited = self.power_w / max(v, 1e-9)
        adhesion_limited = self.spec.adhesion_limit_n(v, self.load_factor)
        return min(current_limited, power_limited, adhesion_limited)

    def effort_n(self, v_ms: float) -> float:
        """Tractive effort applied when accelerating at full power, in newtons.

        Zero at and above the train's maximum speed: the driver stops
        accelerating there. For the force applied while HOLDING that speed see
        :meth:`holding_force_n`.
        """
        if max(v_ms, 0.0) >= self.spec.max_speed_ms:
            return 0.0
        return self.capability_n(v_ms)

    def adhesion_binds(self, v_ms: float) -> bool:
        """Whether the adhesion ceiling is the active constraint at this speed.

        Expected to be False everywhere for the E235. Worth asserting in tests
        so that a later change to the adhesive mass fraction cannot silently
        turn this into a wheelslip-limited model without anyone noticing.
        """
        v = max(v_ms, 0.0)
        if v <= self.base_speed_ms:
            unconstrained = self.max_effort_n
        else:
            unconstrained = self.power_w / max(v, 1e-9)
        return self.spec.adhesion_limit_n(v, self.load_factor) < unconstrained

    def holding_force_n(self, v_ms: float, *, grade_permille: float = 0.0,
                        curve_radius_m: float | None = None) -> float:
        """Tractive effort needed to hold a constant speed, in newtons.

        During cruise the driver throttles back to balance resistance exactly,
        so the applied force is NOT the maximum available at that speed. This
        distinction does not affect run time -- cruise holds speed by
        construction -- but it matters entirely for energy accounting.

        Note in particular that :meth:`effort_n` returns zero at the train's
        maximum speed, which is exactly where cruise happens on this line.
        Recording that as the applied force would credit the train with holding
        90 km/h against 12.8 kN of resistance for free.

        Capped at :meth:`capability_n`, so a speed the train could not sustain
        does not silently produce a force it cannot generate.
        """
        from .resistance import curve_resistance, grade_resistance

        mass = self.spec.mass_kg(self.load_factor)
        required = (
            self.davis(v_ms)
            + grade_resistance(mass, grade_permille)
            + curve_resistance(mass, curve_radius_m)
        )
        return min(max(required, 0.0), self.capability_n(v_ms))

    def net_force_n(self, v_ms: float, *, grade_permille: float = 0.0,
                    curve_radius_m: float | None = None,
                    powering: bool = True) -> float:
        """Net accelerating force, in newtons. Positive accelerates the train."""
        from .resistance import curve_resistance, grade_resistance

        mass = self.spec.mass_kg(self.load_factor)
        traction = self.effort_n(v_ms) if powering else 0.0
        return (
            traction
            - self.davis(v_ms)
            - grade_resistance(mass, grade_permille)
            - curve_resistance(mass, curve_radius_m)
        )

    def acceleration_ms2(self, v_ms: float, **kwargs) -> float:
        """Instantaneous acceleration, in m/s^2, using effective mass."""
        return self.net_force_n(v_ms, **kwargs) / self.spec.effective_mass_kg(self.load_factor)
