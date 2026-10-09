# -*- coding: utf-8 -*-
"""BFS-driven kinetics from reactive MD trajectories.

A compact reference implementation of the workflow presented in the paper:
partial-RDF cutoffs, coordination-shell BFS identification, critical-size
determination, regime-specific kinetic models and the non-isothermal extension.
"""
from .io_dump import Frame, DEFAULT_ELEMENTS, iter_dump, read_frames
from .rdf import partial_rdf, average_partial_rdf, first_minimum, determine_cutoffs
from .structure import (build_bond_graph, coordination_shells, structure_string,
                        canonical_formula, connected_components, identify)
from .events import classify_transition, event_series, coalescence_fragmentation_ratio
from .critical import (gyration_radius, fractal_breakpoint, wilson_interval,
                       breakup_by_size, survival_crossing)
from .kinetics import (bateman, convolve_first_order, aggregation_fragmentation,
                       moment_closure, kernel_from_1_over_m0, nonisothermal_rate,
                       critical_radius, critical_size, nucleation_barrier,
                       observed_nucleation_rate)

__all__ = [
    "Frame", "DEFAULT_ELEMENTS", "iter_dump", "read_frames",
    "partial_rdf", "average_partial_rdf", "first_minimum", "determine_cutoffs",
    "build_bond_graph", "coordination_shells", "structure_string",
    "canonical_formula", "connected_components", "identify",
    "classify_transition", "event_series", "coalescence_fragmentation_ratio",
    "gyration_radius", "fractal_breakpoint", "wilson_interval",
    "breakup_by_size", "survival_crossing",
    "bateman", "convolve_first_order", "aggregation_fragmentation",
    "moment_closure", "kernel_from_1_over_m0", "nonisothermal_rate",
    "critical_radius", "critical_size", "nucleation_barrier",
    "observed_nucleation_rate",
]
