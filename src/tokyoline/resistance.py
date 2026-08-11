"""Running, grade and curve resistance.

The Davis coefficients here are MODELLED, not measured. Published per-class
coefficient sets exist inside Japanese operators but are not released, and no
credible published set for the E235 could be found. What follows is an estimate
built from mass, car count and frontal area, with each physical contribution
named separately so the assumption is auditable rather than a magic number.

This is the largest single uncertainty in the physics core. Any agreement
between simulation and timetable is conditional on it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .units import G, RHO_AIR


@dataclass(frozen=True)
class DavisCoefficients:
    """Running resistance as R(v) = A + B*v + C*v^2, with v in m/s, R in newtons.

    Note the units: railway literature usually writes Davis in km/h and kgf/t.
    Everything here is SI so it can be dropped straight into F = ma.

    Attributes
    ----------
    A : journal, bearing and rolling resistance. Roughly proportional to mass,
        dominant at low speed, and the only term that survives at rest.
    B : flange contact and track interaction. Linear in speed.
    C : aerodynamic drag. Proportional to frontal area, and dominant above
        roughly 80 km/h.
    """

    A: float
    B: float
    C: float
    provenance: str = "modelled"

    def __call__(self, v_ms: float) -> float:
        """Resistance in newtons at speed `v_ms`. Always non-negative."""
        v = abs(v_ms)
        return self.A + self.B * v + self.C * v * v

    def specific_n_per_tonne(self, v_ms: float, mass_kg: float) -> float:
        """Resistance per tonne, the form railway sources usually quote.

        Useful for sanity-checking the estimate: a commuter EMU on level track
        should land around 10-15 N/t at rest and 25-35 N/t at 90 km/h.
        """
        return self(v_ms) / (mass_kg / 1000.0)

    @classmethod
    def estimate_for_emu(
        cls,
        mass_kg: float,
        n_cars: int,
        frontal_area_m2: float,
        *,
        journal_n_per_tonne: float = 10.0,
        flange_n_per_tonne_per_kmh: float = 0.06,
        cd_ends: float = 0.80,
        cd_per_car: float = 0.06,
    ) -> "DavisCoefficients":
        """Estimate Davis coefficients for a multiple unit.

        Each keyword is a separate physical claim, so each can be challenged
        independently:

        journal_n_per_tonne
            Rolling and bearing resistance at rest. Modern roller-bearing stock
            sits around 8-15 N/t; 10 is taken as a central value.
        flange_n_per_tonne_per_kmh
            Flange and track interaction, expressed per km/h because that is how
            it is usually tabulated, then converted here.
        cd_ends
            Combined pressure drag of nose and tail. A commuter EMU has a much
            blunter nose than a Shinkansen, hence a value near 0.8 rather than
            0.2.
        cd_per_car
            Skin friction increment per car. On an 11-car train this roughly
            doubles the total drag coefficient, which is the reason a long train
            cannot be modelled as a single bluff body.

        For tunnel sections the aerodynamic term is substantially higher. That
        is irrelevant for the Yamanote, which is overwhelmingly surface running,
        but would become load-bearing for a Shinkansen extension.
        """
        mass_t = mass_kg / 1000.0

        A = journal_n_per_tonne * mass_t

        # Convert the per-km/h figure into per-(m/s).
        B = flange_n_per_tonne_per_kmh * mass_t * 3.6

        cd_effective = cd_ends + cd_per_car * n_cars
        C = 0.5 * RHO_AIR * cd_effective * frontal_area_m2

        return cls(
            A=A,
            B=B,
            C=C,
            provenance=(
                f"modelled: journal={journal_n_per_tonne} N/t, "
                f"flange={flange_n_per_tonne_per_kmh} N/t/(km/h), "
                f"Cd_eff={cd_effective:.2f} over {frontal_area_m2:.2f} m^2 "
                f"({n_cars} cars). No published E235 coefficient set available."
            ),
        )


def grade_resistance(mass_kg: float, grade_permille: float) -> float:
    """Gravitational resistance on a gradient, in newtons.

    Japanese practice expresses gradient in per-mille, so for the small angles
    involved sin(theta) is grade/1000 to well within any other error in the
    model. Positive grade means uphill and therefore positive (retarding) force.

    On the Yamanote this term is negligible -- the line is close to flat, and
    per-segment gradient data is not public at usable precision anyway. It is
    implemented for correctness and so the model generalises to lines where it
    matters.
    """
    return mass_kg * G * (grade_permille / 1000.0)


def curve_resistance(mass_kg: float, radius_m: float | None,
                     coefficient: float = 600.0) -> float:
    """Additional resistance on a curve, in newtons.

    Empirical, of the form k / R. Several formulations with different constants
    are in use; `coefficient` is in N*m per tonne and 600 is a common value for
    standard gauge.

    Returns zero when `radius_m` is None, which is the honest default here:
    per-segment curve radius data for the Yamanote is not available, so this
    term is carried in the code and evaluated at zero. It is a small effect.
    """
    if radius_m is None or radius_m <= 0.0:
        return 0.0
    return coefficient * (mass_kg / 1000.0) / radius_m
