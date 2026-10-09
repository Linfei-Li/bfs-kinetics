# -*- coding: utf-8 -*-
"""Bond-graph construction, coordination-shell BFS and molecular identification.

For every frame a bond graph is built with the element-pair cutoffs obtained
from the partial RDF. Molecules and clusters are the connected components of
that graph. The coordination environment of any atom is enumerated shell by
shell with breadth-first search; edges leading to an already visited atom are
skipped so that rings are not counted twice.
"""
from __future__ import annotations
from collections import defaultdict, deque
import numpy as np

from .io_dump import Frame, DEFAULT_ELEMENTS


def min_image_delta(a, b, box):
    d = a - b
    d -= box * np.round(d / box)
    return d


def _pair_key(symbol_a, symbol_b):
    return tuple(sorted((symbol_a, symbol_b)))


def build_bond_graph(frame: Frame, cutoffs, element_map=DEFAULT_ELEMENTS):
    """Return adjacency as ``{i: [(j, distance), ...]}``.

    ``cutoffs`` maps an element-pair tuple such as ``("A", "B")`` to a distance
    in Angstrom; ordering of the pair is irrelevant.
    """
    sym = frame.symbols(element_map)
    pair_cut = {_pair_key(a, b): float(c) for (a, b), c in cutoffs.items()}
    r_max = max(pair_cut.values())
    cell = r_max
    nx = max(1, int(np.floor(frame.box[0] / cell)))
    ny = max(1, int(np.floor(frame.box[1] / cell)))
    nz = max(1, int(np.floor(frame.box[2] / cell)))
    actual = np.array([frame.box[0] / nx, frame.box[1] / ny, frame.box[2] / nz])

    grid = defaultdict(list)
    for i, p in enumerate(frame.pos):
        grid[(int(p[0] / actual[0]) % nx,
              int(p[1] / actual[1]) % ny,
              int(p[2] / actual[2]) % nz)].append(i)

    adj = defaultdict(list)
    for (cx, cy, cz), members in grid.items():
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    other = ((cx + dx) % nx, (cy + dy) % ny, (cz + dz) % nz)
                    for i in members:
                        for j in grid.get(other, []):
                            if i >= j:
                                continue
                            key = _pair_key(sym[i], sym[j])
                            rc = pair_cut.get(key)
                            if rc is None:
                                continue
                            d = float(np.linalg.norm(
                                min_image_delta(frame.pos[i], frame.pos[j], frame.box)))
                            if d <= rc:
                                adj[i].append((j, d))
                                adj[j].append((i, d))
    return adj


def coordination_shells(adj, seed):
    """Breadth-first enumeration around ``seed``.

    Returns
    -------
    shells : list[list[int]]
        shell 0 is the seed, shell 1 its bonded neighbours, and so on.
    parent : dict[int, int]
    cross_edges : list[(int, int)]
        Bonds that reach an already visited atom (ring closures); these are the
        neighbours that the search skips.
    """
    shells = [[seed]]
    parent = {seed: seed}
    cross_edges = []
    frontier = [seed]
    while frontier:
        nxt = []
        for node in frontier:
            for nb, _ in adj.get(node, []):
                if nb not in parent:
                    parent[nb] = node
                    nxt.append(nb)
                elif parent.get(node) != nb and parent.get(nb) != node:
                    pair = tuple(sorted((node, nb)))
                    if pair not in [tuple(sorted(e)) for e in cross_edges]:
                        cross_edges.append((node, nb))
        if not nxt:
            break
        shells.append(nxt)
        frontier = nxt
    return shells, parent, cross_edges


def structure_string(seed, adj, frame: Frame, element_map=DEFAULT_ELEMENTS):
    """Canonical nested structure string rooted at ``seed``.

    Example, a central atom bonded to one bridging and two terminal atoms::

        A6[B2(1.85),C4(2.24),C9(2.26)]

    Ring-closing bonds are omitted, matching the visited-atom skip in the search.
    """
    sym = frame.symbols(element_map)
    _, parent, _ = coordination_shells(adj, seed)
    children = defaultdict(list)
    for node, par in parent.items():
        if par != node:
            children[par].append(node)

    def order(ns):
        return sorted(ns, key=lambda n: (sym[n], frame.ids[n]))

    def render(node):
        label = f"{sym[node]}{frame.ids[node]}"
        kids = order(children.get(node, []))
        if not kids:
            return label
        parts = []
        for k in kids:
            d = np.linalg.norm(min_image_delta(frame.pos[k], frame.pos[node], frame.box))
            parts.append(f"{render(k)}({d:.2f})")
        return f"{label}[{','.join(parts)}]"

    return render(seed)


def canonical_formula(indices, frame: Frame, element_map=DEFAULT_ELEMENTS,
                      central_element=None):
    """Hill-like formula. The central element (if given and present) is listed
    first, followed by the remaining elements in alphabetical order."""
    sym = frame.symbols(element_map)
    counts = defaultdict(int)
    for i in indices:
        counts[sym[i]] += 1
    head = [central_element] if central_element and counts.get(central_element) else []
    tail = sorted(s for s in counts if s not in head)
    out = ""
    for s in head + tail:
        out += s + (str(counts[s]) if counts[s] > 1 else "")
    return out


def connected_components(adj, n_atoms):
    """All connected components as a list of index sets, including isolated atoms."""
    seen = set()
    components = []
    for start in range(n_atoms):
        if start in seen:
            continue
        comp = set()
        queue = deque([start])
        seen.add(start)
        while queue:
            node = queue.popleft()
            comp.add(node)
            for nb, _ in adj.get(node, []):
                if nb not in seen:
                    seen.add(nb)
                    queue.append(nb)
        components.append(comp)
    return components


def identify(frame: Frame, cutoffs, element_map=DEFAULT_ELEMENTS,
             central_element=None):
    """Identify every molecule/cluster in a frame.

    ``central_element`` is the element whose count defines the cluster size and
    ordering. Returns a list of dictionaries with keys ``indices``, ``formula``,
    ``n_central`` and ``n_atoms``, sorted by decreasing central-element count.
    """
    adj = build_bond_graph(frame, cutoffs, element_map)
    sym = frame.symbols(element_map)
    out = []
    for comp in connected_components(adj, frame.n_atoms):
        n_central = int(sum(1 for i in comp
                            if central_element and sym[i] == central_element))
        out.append({
            "indices": comp,
            "formula": canonical_formula(comp, frame, element_map, central_element),
            "n_central": n_central,
            "n_atoms": len(comp),
        })
    out.sort(key=lambda d: (-d["n_central"], d["formula"]))
    return out, adj
