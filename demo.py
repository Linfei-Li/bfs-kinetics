# -*- coding: utf-8 -*-
"""
demo.py
-------
End-to-end demonstration and self-test of the ``bfs_kinetics`` package on a
small *synthetic* gas-phase trajectory. No external data are required.

The example uses generic element labels A (central), B (bridging) and C
(terminal), so it is not tied to any chemical system. It simulates the
stochastic, reversible aggregation of ABC2 monomers with a planted critical
size (clusters at or below it may break, larger clusters do not), generates
3-D coordinates consistent with a subcritical and a denser supercritical
fractal dimension, and then runs the full workflow:

  RDF-derived cutoffs -> BFS identification -> Rg power-law breakpoint
  -> frame-to-frame event classification with exact reconciliation
  -> constant-kernel 1/M0 linearity -> T**1/2 scaling -> CNT check.

Run:  python demo.py
A final "ALL CHECKS PASSED" line means every invariant held.
"""
import numpy as np
from bfs_kinetics import (Frame, determine_cutoffs, identify, event_series,
                          fractal_breakpoint, kernel_from_1_over_m0,
                          nonisothermal_rate, critical_radius, critical_size,
                          nucleation_barrier, observed_nucleation_rate)
from bfs_kinetics.structure import min_image_delta

rng = np.random.default_rng(7)

# --- generic system: N monomers of ABC2 (1 B + 2 C each) -------------------
N = 20
NSTAR = 4                       # planted critical size
BOX = np.array([42.0, 42.0, 42.0])

# global ids: A 1..20, B 21..40, C 41..80
ELEMENTS = {1: "A", 2: "B", 3: "C"}
CENTRAL = "A"
SYM2TYPE = {s: t for t, s in ELEMENTS.items()}
TYP = np.zeros(81, dtype=int)
for i in range(1, 21):
    TYP[i] = SYM2TYPE["A"]
    TYP[20 + i] = SYM2TYPE["B"]
    TYP[2 * i + 39] = SYM2TYPE["C"]
    TYP[2 * i + 40] = SYM2TYPE["C"]


# --- stochastic aggregation with a planted critical size -------------------
clusters = {i: {i} for i in range(N)}
next_id = N
snapshots = []
record_steps = list(range(0, 501, 50))
record_at = set(record_steps)
p_pair, p_break = 0.008, 0.10

for step in range(501):
    if step in record_at:
        snapshots.append([frozenset(s) for s in clusters.values()])
    cids = list(clusters.keys())
    pairs = [(cids[i], cids[j]) for i in range(len(cids))
             for j in range(i + 1, len(cids))]
    rng.shuffle(pairs)
    for a, b in pairs:                              # per-pair coalescence
        if a not in clusters or b not in clusters:
            continue
        if rng.random() < p_pair:
            clusters[a] |= clusters[b]
            del clusters[b]
    for cid in list(clusters):                        # binary breakup (subcrit)
        s = clusters[cid]
        if 2 <= len(s) <= NSTAR and rng.random() < p_break:
            members = list(s)
            rng.shuffle(members)
            cut = int(rng.integers(1, len(members)))
            clusters[cid] = set(members[:cut])
            clusters[next_id] = set(members[cut:])
            next_id += 1


# --- geometry generation ---------------------------------------------------
def random_unit():
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def unit_perp(e):
    v = random_unit()
    v -= np.dot(v, e) * e
    return v / np.linalg.norm(v)


def build_local_cluster(members):
    """Build a connected cluster in coordinates relative to its first A atom.

    A spanning tree of A atoms is joined by shared bridging B atoms (A-B-A).
    Subcritical clusters extend as a chain, supercritical clusters branch and
    compact. Each A keeps two terminal C and the root keeps a terminal B.
    Returns (relative positions id->xyz, occupied radius).
    """
    members = sorted(members)
    n = len(members)

    placed = [members[0]]
    parents = {}
    for m in members[1:]:
        parents[m] = placed[-1] if n <= NSTAR else int(rng.choice(placed))
        placed.append(m)

    a_pos = {members[0]: np.zeros(3)}
    bridge = {}
    for m in members[1:]:
        rp = a_pos[parents[m]]
        e1 = random_unit()
        bo = rp + e1 * rng.uniform(1.80, 1.95)
        theta = np.deg2rad(rng.uniform(120.0, 150.0))
        e2 = -np.cos(theta) * e1 + np.sin(theta) * unit_perp(e1)
        a_pos[m] = bo + e2 * rng.uniform(1.80, 1.95)
        bridge[m] = bo

    rel = {}
    root = members[0]
    for m, p in a_pos.items():
        tid = m + 1
        rel[tid] = p
        for c_id in (2 * tid + 39, 2 * tid + 40):
            rel[c_id] = p + random_unit() * rng.uniform(2.15, 2.40)
    rel[20 + root + 1] = a_pos[root] + random_unit() * rng.uniform(1.75, 1.95)
    for m in members[1:]:
        rel[20 + m + 1] = bridge[m]

    occupied = max(float(np.linalg.norm(v)) for v in rel.values())
    return rel, occupied


def pack_centers(occupied):
    """Place cluster centers by minimum-image rejection so distinct clusters
    stay separated by more than any bond cutoff."""
    order = sorted(range(len(occupied)), key=lambda k: -occupied[k])
    centers = {}
    for k in order:
        for _attempt in range(80000):
            p = rng.uniform(0.0, BOX, size=3)
            if all(np.linalg.norm(min_image_delta(p, centers[j], BOX))
                   >= occupied[k] + occupied[j] + 3.2 for j in centers):
                centers[k] = p
                break
        else:
            raise RuntimeError("Could not pack clusters without overlap.")
    return centers


def build_frame(step, partition):
    locals_ = [build_local_cluster(s) for s in partition]
    occupied = [oc for _rel, oc in locals_]
    centers = pack_centers(occupied)
    pos = {}
    for k, (rel, _oc) in enumerate(locals_):
        for aid, v in rel.items():
            pos[aid] = centers[k] + v
    arr = np.array([pos[a] for a in range(1, 81)], dtype=float)
    arr = arr - BOX * np.floor(arr / BOX)
    return Frame(step, BOX, np.arange(1, 81), TYP[1:], arr)


frames = [build_frame(step, part) for step, part in zip(record_steps, snapshots)]
print(f"Generated {len(frames)} frames; final cluster count = {len(snapshots[-1])}")

# --- 1. data-driven cutoffs ------------------------------------------------
cutoffs = determine_cutoffs(frames[2:], pairs=[("A", "B"), ("A", "C")],
                            r_max=6.0, nbins=120, element_map=ELEMENTS)
rc_ab = cutoffs[("A", "B")]
rc_ac = cutoffs[("A", "C")]
print(f"RDF first-minimum cutoffs: A-B = {rc_ab:.2f} A, A-C = {rc_ac:.2f} A")
assert 2.0 < rc_ab < 3.3 and 2.5 < rc_ac < 3.6

# --- 2. BFS identification must recover planted clusters -------------------
identified = []
for f, frame in enumerate(frames):
    out, _adj = identify(frame, cutoffs, ELEMENTS, central_element=CENTRAL)
    identified.append(out)
    found = set()
    for comp in out:
        a_grp = frozenset(int(frame.ids[i]) for i in comp["indices"]
                          if frame.types[i] == SYM2TYPE["A"])
        if a_grp:
            found.add(a_grp)
    planted_global = {frozenset(m + 1 for m in s) for s in snapshots[f]}
    assert found == planted_global, (
        f"frame {f}: components mismatch\n"
        f"  found A groups: {sorted((sorted(t) for t in found))}\n"
        f"  planted:        {sorted((sorted(t) for t in planted_global))}")
print("BFS components reproduce the planted clusters on every frame.")

# --- 3. Rg power-law breakpoint (planted two-slope data) ------------------
sizes = np.arange(1, 21)
planted_rg = []
for n in sizes:
    df = 1.7 if n <= NSTAR else 2.7
    planted_rg.append(2.0 * n ** (1.0 / df))
planted_rg = np.array(planted_rg) * rng.lognormal(0.0, 0.03, size=sizes.size)
n_star, df_below, df_above, _res = fractal_breakpoint(sizes, planted_rg)
print(f"Rg breakpoint at n = {n_star}; fractal dimensions "
      f"{df_below:.2f} -> {df_above:.2f}")
assert n_star in (NSTAR, NSTAR + 1)
assert abs(df_below - 1.7) < 0.15 and abs(df_above - 2.7) < 0.15

# --- 4. event classification and exact reconciliation ----------------------
series = event_series(identified)
for i, r in enumerate(series, start=1):
    rec = r["reconciliation"]
    assert rec["balanced"], f"frame {i}: events do not reconcile"
n_add = sum(len(r["additions"]) for r in series)
n_coal = sum(len(r["coalescences"]) for r in series)
n_frag = sum(len(r["fragmentations"]) for r in series)
print(f"Events over trajectory: {n_add} additions, {n_coal} coalescences, "
      f"{n_frag} fragmentations; counts reconcile exactly.")

# --- 5. constant-kernel 1/M0 linearity (analytic M0 with noise) -----------
V_box, K_coal, M0_init = 1.0, 0.5, 40.0
t_axis = np.linspace(0.0, 5.0, 21)
m0_analytic = M0_init / (1.0 + 0.5 * K_coal * M0_init / V_box * t_axis)
m0_analytic *= rng.lognormal(0.0, 0.01, size=m0_analytic.size)
k_fit, r2, _a, _b = kernel_from_1_over_m0(t_axis, m0_analytic, V_box)
print(f"1/M0 linearity R^2 = {r2:.4f}; recovered k_coal = {k_fit:.3f} "
      f"(true {K_coal})")
assert r2 > 0.99 and abs(k_fit - K_coal) < 0.05

# --- 6. T**1/2 scaling -----------------------------------------------------
k4 = nonisothermal_rate(4000.0, 1000.0, 1.0)
assert abs(k4 - np.sqrt(4.0)) < 1e-12
print("Gas-kinetic scaling gives k(4T)/k(T) = sqrt(4) = 2.")

# --- 7. classical nucleation theory (consistent eV / A units) --------------
gamma = 0.38 / 16.02                 # representative surface tension, eV/A^2
v0 = 30.0                            # A^3 per formula unit
temp = 2000.0
ln_s = 2.3
r_star = critical_radius(gamma, v0, temp, ln_s)
i_star = critical_size(r_star, v0)
barrier = nucleation_barrier(i_star, ln_s)
print(f"CNT: r* = {r_star:.2f} A, i* = {i_star:.1f}, dG* = {barrier:.1f} kT")
assert 6 <= i_star <= 12 and 7 <= barrier <= 13

# --- observed nucleation rate (exercise the helper) ------------------------
vol_cm3 = (42.0e-8) ** 3
j_obs = observed_nucleation_rate(vol_cm3, 5.2e-9)
print(f"Example J_obs for the demo box = {j_obs:.2e} cm^-3 s^-1")

print("\nALL CHECKS PASSED")
