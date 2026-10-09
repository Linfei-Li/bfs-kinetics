# -*- coding: utf-8 -*-
"""Partial radial distribution functions and the data-driven bond cutoff.

The bond cutoff for each element pair is the first minimum of the corresponding
partial RDF. No external bond length or force-field bond order is used, and the
same cutoff is shared by molecular and cluster identification.
"""
from __future__ import annotations
import numpy as np

from .io_dump import DEFAULT_ELEMENTS


def _pair_distances(frame, type_a: int, type_b: int, box=None):
    """Minimum-image distances between atoms of two types."""
    box = frame.box if box is None else np.asarray(box, dtype=float)
    ia = np.where(frame.types == type_a)[0]
    ib = np.where(frame.types == type_b)[0]
    if type_a == type_b:
        if len(ia) < 2:
            return np.empty(0)
        a, b = np.triu_indices(len(ia), k=1)
        ii, jj = ia[a], ia[b]
    else:
        if len(ia) == 0 or len(ib) == 0:
            return np.empty(0)
        ii, jj = np.meshgrid(ia, ib, indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
    d = frame.pos[ii] - frame.pos[jj]
    d -= box * np.round(d / box)
    return np.sqrt(np.einsum("ij,ij->i", d, d))


def partial_rdf(frame, symbol_a: str, symbol_b: str,
                r_max: float | None = None, nbins: int = 300,
                element_map=DEFAULT_ELEMENTS):
    """Return (r, g_ab(r)) for one element pair.

    Symbols are element names (e.g. ``"A"``, ``"B"``).
    """
    sym_to_type = {s: t for t, s in element_map.items()}
    ta, tb = sym_to_type[symbol_a], sym_to_type[symbol_b]
    if r_max is None:
        r_max = float(frame.box.min()) / 2.0
    edges = np.linspace(0.0, r_max, nbins + 1)
    dist = _pair_distances(frame, ta, tb)
    hist, _ = np.histogram(dist, bins=edges)
    r = 0.5 * (edges[:-1] + edges[1:])
    shell = 4.0 / 3.0 * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    volume = float(np.prod(frame.box))
    na = int(np.sum(frame.types == ta))
    nb = int(np.sum(frame.types == tb))
    if ta == tb:
        norm = 2.0 * volume / max(na * (na - 1), 1)
    else:
        norm = volume / max(na * nb, 1)
    g = hist.astype(float) * norm / shell
    return r, g


def average_partial_rdf(frames, symbol_a, symbol_b,
                        r_max=None, nbins=300, element_map=DEFAULT_ELEMENTS):
    """Average a partial RDF over many frames."""
    if r_max is None:
        r_max = float(min(fr.box.min() for fr in frames)) / 2.0
    edges = np.linspace(0.0, r_max, nbins + 1)
    r = 0.5 * (edges[:-1] + edges[1:])
    acc = np.zeros(nbins)
    for fr in frames:
        rr, gg = partial_rdf(fr, symbol_a, symbol_b, r_max=r_max,
                             nbins=nbins, element_map=element_map)
        acc += gg
    return r, acc / max(len(frames), 1)


def _smooth(y, window=5):
    if window <= 1:
        return y
    kernel = np.ones(window) / window
    pad = window // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    return np.convolve(yp, kernel, mode="valid")


def first_minimum(r, g, search_frac: float = 0.55, smooth_window: int = 7,
                  peak_min_frac: float = 0.12, peak_contrast: float = 0.2):
    """Locate the first minimum after the first coordination peak.

    The peak is the tallest maximum between ``peak_min_frac`` and
    ``search_frac`` of the range; the lower bound excludes unphysical
    near-zero bins that a single very close pair would populate. The minimum
    is the first subsequent point where g stops decreasing. The result is
    insensitive to a small shift of the binning.

    If there is no distinct coordination peak (the peak-to-valley contrast is
    below ``peak_contrast``), the pair does not define a bond shell and the
    function returns ``None``.
    """
    r = np.asarray(r)
    gs = _smooth(np.asarray(g), smooth_window)
    start_bin = int(peak_min_frac * len(r))
    cutoff_bin = int(search_frac * len(r))
    peak = start_bin + int(np.argmax(gs[start_bin:cutoff_bin]))
    i = peak
    tol = 1e-3 * max(gs[peak], 1.0)
    while i + 1 < len(r) - 1 and gs[i + 1] <= gs[i] + tol:
        i += 1
    # refine: take the lowest point within +/-1 bin of the turning location
    lo, hi = max(peak + 1, i - 1), min(len(r), i + 2)
    i = lo + int(np.argmin(gs[lo:hi]))
    contrast = (gs[peak] - gs[i]) / max(abs(gs[peak]), 1e-9)
    if contrast < peak_contrast:
        return None
    return float(r[i])


def determine_cutoffs(frames, pairs=None,
                      r_max=None, nbins=300, element_map=DEFAULT_ELEMENTS):
    """Build the ``{(a, b): r_c, ...}`` cutoff dictionary from the trajectory.

    If ``pairs`` is not given, a cutoff is determined for every element pair
    that appears in the frames; pairs without a distinct coordination peak are
    skipped.
    """
    if pairs is None:
        symbols = set()
        for fr in frames:
            symbols.update(fr.symbols(element_map))
        symbols = sorted(symbols)
        pairs = [(a, b) for i, a in enumerate(symbols)
                 for b in symbols[i:]]
    cutoffs = {}
    for a, b in pairs:
        r, g = average_partial_rdf(frames, a, b, r_max=r_max,
                                   nbins=nbins, element_map=element_map)
        rc = first_minimum(r, g)
        if rc is not None:
            cutoffs[(a, b)] = rc
    return cutoffs
