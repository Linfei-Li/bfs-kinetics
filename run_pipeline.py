# -*- coding: utf-8 -*-
"""
run_pipeline.py
---------------
Command-line pipeline that turns a text LAMMPS dump trajectory into
data-driven bond cutoffs, per-frame molecular formulas and cluster sizes.

Example
-------
python run_pipeline.py --dump trajectory.dump --out results/ \
        --element-map "1=A,2=B,3=C" --central-element A \
        --pairs "A-B,A-C" --r-max 6.0 --nbins 120

Outputs (written to --out)
--------------------------
cutoffs.csv         bond cutoff per element pair (RDF first minimum)
formula_counts.csv  timestep, formula, count (long format)
cluster_sizes.csv   timestep, number of central-element clusters, sizes
"""
import argparse
import csv
import os
from collections import Counter

from bfs_kinetics import read_frames, determine_cutoffs, identify


def parse_element_map(text):
    """Parse ``1=A,2=B,3=C`` into ``{1: "A", 2: "B", 3: "C"}``."""
    mapping = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        key, val = item.split("=")
        mapping[int(key.strip())] = val.strip()
    return mapping


def parse_pairs(text):
    """Parse ``A-B,A-C`` into ``[("A", "B"), ("A", "C")]``."""
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if item:
            a, b = item.split("-")
            pairs.append((a.strip(), b.strip()))
    return pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", required=True, help="path to the LAMMPS dump file")
    ap.add_argument("--out", required=True, help="directory for output CSV files")
    ap.add_argument("--element-map", required=True,
                    help='LAMMPS type to element map, e.g. "1=A,2=B,3=C"')
    ap.add_argument("--central-element", default=None,
                    help="element whose count defines cluster size")
    ap.add_argument("--pairs", default=None,
                    help='element pairs for cutoffs, e.g. "A-B,A-C"; '
                         "defaults to all pairs in the trajectory")
    ap.add_argument("--step-interval", type=int, default=None,
                    help="keep timesteps divisible by this value")
    ap.add_argument("--r-max", type=float, default=6.0)
    ap.add_argument("--nbins", type=int, default=120)
    args = ap.parse_args()

    element_map = parse_element_map(args.element_map)
    pairs = parse_pairs(args.pairs) if args.pairs else None
    central = args.central_element

    os.makedirs(args.out, exist_ok=True)
    frames = read_frames(args.dump, args.step_interval)
    if len(frames) < 2:
        raise SystemExit("Need at least two frames to determine cutoffs.")
    print(f"Read {len(frames)} frames from {args.dump}")

    cutoffs = determine_cutoffs(frames, pairs=pairs, r_max=args.r_max,
                                nbins=args.nbins, element_map=element_map)
    with open(os.path.join(args.out, "cutoffs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "cutoff_A"])
        for pair, rc in sorted(cutoffs.items()):
            w.writerow([f"{pair[0]}-{pair[1]}", f"{rc:.4f}"])
    print("Cutoffs:", {f"{a}-{b}": round(rc, 3) for (a, b), rc in cutoffs.items()})

    formula_rows = []
    size_rows = []
    for frame in frames:
        components, _adj = identify(frame, cutoffs, element_map,
                                    central_element=central)
        counts = Counter(c["formula"] for c in components)
        for formula, count in sorted(counts.items()):
            formula_rows.append((frame.timestep, formula, count))
        sizes = sorted((c["n_central"] for c in components
                        if c["n_central"] > 0), reverse=True)
        size_rows.append((frame.timestep, len(sizes),
                          " ".join(str(n) for n in sizes)))

    with open(os.path.join(args.out, "formula_counts.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestep", "formula", "count"])
        w.writerows(formula_rows)

    with open(os.path.join(args.out, "cluster_sizes.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestep", "n_clusters", "central_sizes"])
        w.writerows(size_rows)

    print(f"Wrote {len(formula_rows)} formula rows and {len(size_rows)} "
          f"cluster-size rows to {args.out}")


if __name__ == "__main__":
    main()
