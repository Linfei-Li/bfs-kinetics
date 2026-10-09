# -*- coding: utf-8 -*-
"""Event classification between consecutive frames.

Components are matched across two frames by the atoms they share. A new
component that receives atoms from several earlier components is a merge; it is
monomer addition (n+1) when a single cluster absorbs one central-atom species
and coalescence (a+b) when two multi-atom clusters join. An earlier component
that feeds several new components is a fragmentation. Atom conservation makes
the component-count change reconcile exactly with these events.
"""
from __future__ import annotations
from collections import defaultdict


def _key(comp):
    return frozenset(comp["indices"])


def classify_transition(old_comps, new_comps):
    """Classify the events between two frames.

    Returns a dict with lists ``additions``, ``coalescences``, ``fragmentations``
    and a ``reconciliation`` block comparing predicted and actual changes of the
    number of central-atom-bearing components.
    """
    old = {_key(c): c for c in old_comps}
    new = {_key(c): c for c in new_comps}

    # parents of each new component, children of each old component
    parents = defaultdict(list)
    children = defaultdict(list)
    for ok, oc in old.items():
        for nk, nc in new.items():
            inter = len(ok & nk)
            if inter:
                parents[nk].append((ok, inter))
                children[ok].append((nk, inter))

    additions, coalescences, fragmentations = [], [], []
    for nk, plist in parents.items():
        sizes = sorted((old[k]["n_central"] for k, _ in plist), reverse=True)
        sizes = [s for s in sizes if s > 0]
        if len(plist) >= 2 and len(sizes) >= 2:
            event = {"sizes": sizes, "result": new[nk]["n_central"],
                     "formula": new[nk]["formula"]}
            if sizes[0] >= 2 and all(s == 1 for s in sizes[1:]):
                additions.append(event)
            else:
                coalescences.append(event)

    for ok, clist in children.items():
        if len(clist) >= 2:
            frag_sizes = sorted((new[k]["n_central"] for k, _ in clist),
                                reverse=True)
            if old[ok]["n_central"] >= 1 and sum(frag_sizes) >= 1:
                fragmentations.append({
                    "size": old[ok]["n_central"],
                    "fragments": frag_sizes,
                    "formula": old[ok]["formula"],
                })

    # reconciliation of the number of central-atom-bearing components
    n_old = sum(1 for c in old_comps if c["n_central"] > 0)
    n_new = sum(1 for c in new_comps if c["n_central"] > 0)
    merge_change = sum(len([k for k, _ in plist if old[k]["n_central"] > 0]) - 1
                       for nk, plist in parents.items() if len(plist) >= 2)
    frag_change = sum(len([k for k, _ in clist if new[k]["n_central"] > 0]) - 1
                      for ok, clist in children.items() if len(clist) >= 2)
    predicted = n_old - merge_change + frag_change

    return {
        "additions": additions,
        "coalescences": coalescences,
        "fragmentations": fragmentations,
        "reconciliation": {
            "n_old": n_old,
            "n_new": n_new,
            "predicted": predicted,
            "balanced": predicted == n_new,
        },
    }


def event_series(identified_frames):
    """Run :func:`classify_transition` over a time-ordered list of frame components."""
    results = []
    for i in range(1, len(identified_frames)):
        results.append(classify_transition(identified_frames[i - 1],
                                          identified_frames[i]))
    return results


def coalescence_fragmentation_ratio(results, bin_size: int = 1):
    """Aggregate (coalescence+addition)/fragmentation counts, optionally binned."""
    bins = defaultdict(lambda: {"coalescence": 0, "fragmentation": 0})
    for i, r in enumerate(results):
        b = (i // bin_size) * bin_size
        bins[b]["coalescence"] += len(r["coalescences"]) + len(r["additions"])
        bins[b]["fragmentation"] += len(r["fragmentations"])
    out = []
    for b in sorted(bins):
        f = bins[b]["fragmentation"]
        out.append({"bin": b, "coalescence": bins[b]["coalescence"],
                    "fragmentation": f,
                    "ratio": bins[b]["coalescence"] / f if f else float("inf")})
    return out
