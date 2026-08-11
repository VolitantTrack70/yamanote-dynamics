"""Unit constants and conversions.

Japanese rolling stock performance is published in km/h/s, not m/s^2. Mixing the
two silently is the easiest way to be wrong by a factor of 3.6, so every
conversion in this project goes through here and internal state is ALWAYS SI.
"""

#: Standard gravity, m/s^2.
G = 9.80665

#: Air density at sea level, 15 C, kg/m^3.
RHO_AIR = 1.225

#: 1 km/h expressed in m/s.
KMH_TO_MS = 1.0 / 3.6

#: 1 m/s expressed in km/h.
MS_TO_KMH = 3.6

#: 1 km/h/s expressed in m/s^2. Equal to 0.2777... exactly 1/3.6.
KMHS_TO_MS2 = 1.0 / 3.6


def kmh(v_ms: float) -> float:
    """Convert m/s to km/h, for display only."""
    return v_ms * MS_TO_KMH


def ms(v_kmh: float) -> float:
    """Convert km/h to m/s."""
    return v_kmh * KMH_TO_MS


def ms2(a_kmh_s: float) -> float:
    """Convert km/h/s to m/s^2."""
    return a_kmh_s * KMHS_TO_MS2
