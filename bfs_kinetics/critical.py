# -*- coding: utf-8 -*-
"""Critical-size determination without an assumed threshold.

Geometry gives two power laws Rg proportional to n^(1/df) whose residual-minimizing
breakpoint locates the transition; fragmentation statistics give the fraction of
break-up events by size and the survival probability that crosses one half at the
same size. These independent lines of evidence converge on one critical size.
"""
from __future__ import annotations
import numpy as np


def gyration_radius(pos):
    """Radius of gyration about the centre of mass."""
    pos = np.asarray(pos, dtype=float)
    cm = pos.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((pos - cm) ** 2, axis=1))))


def fractal_breakpoint(sizes, rg, min_points=2):
    """Fit two log-log power laws and choose the residual-minimizing breakpoint.

    Returns ``(n_star, df_below, df_above, residuals)`` where df is the fractal
    dimension (df = 1/slope).
    """
    n = np.asarray(sizes, dtype=float)
    r = np.asarray(rg, dtype=float)
    order = np.argsort(n)
    n, r = n[order], r[order]
    x, y = np.log(n), np.log(r)
    best = None
    residuals = {}
    for b in range(min_points, len(n) - min_points + 1):
        s1, res1, *_ = np.linalg.lstsq(
            np.vstack([x[:b], np.ones(b)]).T, y[:b], rcond=None)
        s2, res2, *_ = np.linalg.lstsq(
            np.vstack([x[b:], np.ones(len(x) - b)]).T, y[b:], rcond=None)
        r1 = float(np.sum((y[:b] - (s1[0] * x[:b] + s1[1])) ** 2))
        r2 = float(np.sum((y[b:] - (s2[0] * x[b:] + s2[1])) ** 2))
        total = r1 + r2
        residuals[int(n[b - 1])] = total
        if best is None or total < best[0]:
            best = (total, b, s1[0], s2[0])
    _, b, sl1, sl2 = best
    n_star = int(n[b - 1])
    return n_star, 1.0 / sl1, 1.0 / sl2, residuals


def wilson_interval(k, n, z=1.96):
    """Wilson 95% confidence interval for a binomial proportion k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(centre - half), float(centre + half))


def breakup_by_size(fragmentation_events):
    """Count fragmentation events by parent cluster size -> {size: count}."""
    counts = {}
    for ev in fragmentation_events:
        counts[ev["size"]] = counts.get(ev["size"], 0) + 1
    return counts


def survival_crossing(counts_by_size):
    """Size at which the break-up-derived survival probability crosses 1/2.

    ``pgrow(n)`` is approximated as one minus the cumulative fraction of
    break-up events occurring at sizes below n. Returns the crossing size, the
    cumulative fractions and pgrow values.
    """
    sizes = sorted(counts_by_size)
    total = sum(counts_by_size.values())
    cum = 0
    cumulative, pgrow = {}, {}
    crossing = None
    for s in sizes:
        cumulative[s] = cum / total
        pgrow[s] = 1 - cumulative[s]
        if crossing is None and pgrow[s] >= 0.5 and cumulative[s] <= 0.5:
            crossing = s
        cum += counts_by_size[s]
    # half of break-ups lie at or below the median event size
    return crossing, cumulative, pgrow
