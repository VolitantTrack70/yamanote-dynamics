"""Tokyo line train dynamics and ridership model.

A first-principles physics simulation of a real Tokyo rail line, validated
against the published timetable. The point of the project is not the physics
loop -- it is the quantified, explained gap between what the physics says is
possible and what the railway actually schedules.
"""

from .brake import DEFAULT_JERK_LIMIT, BrakeProfile
from .conditions import CONDITIONS, DEFAULT_CONDITION, RailCondition
from .data import (
    Segment,
    build_model,
    load_congestion,
    load_ridership,
    load_segments,
    load_spec,
    load_stations,
)
from .network import Circuit, LoopGeometry, load_coordinates, simulate_circuit
from .resistance import DavisCoefficients, curve_resistance, grade_resistance
from .segment import Phase, SegmentResult, simulate_segment
from .stock import TrainSpec, adhesion_coefficient
from .traction import TractionModel

__all__ = [
    "BrakeProfile",
    "CONDITIONS",
    "Circuit",
    "DEFAULT_CONDITION",
    "DEFAULT_JERK_LIMIT",
    "DavisCoefficients",
    "LoopGeometry",
    "Phase",
    "RailCondition",
    "Segment",
    "SegmentResult",
    "TrainSpec",
    "TractionModel",
    "adhesion_coefficient",
    "build_model",
    "curve_resistance",
    "grade_resistance",
    "load_congestion",
    "load_coordinates",
    "load_ridership",
    "load_segments",
    "load_spec",
    "load_stations",
    "simulate_circuit",
    "simulate_segment",
]
