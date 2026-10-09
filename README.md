# BFS-driven kinetics from reactive MD trajectories

[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.23258942.svg)](https://doi.org/10.5281/zenodo.23258942)

A compact reference implementation of the data-driven workflow described in
the paper:

1. **Data-driven bond cutoffs.** The cutoff for every element pair is the
   first minimum of the corresponding partial radial distribution function.
   No external bond length or force-field bond order is used, and the same
   cutoff is shared by molecular and cluster identification.
2. **Coordination-shell identification.** A bond graph is built for each
   frame; molecules and clusters are its connected components, enumerated
   with breadth-first search so that ring-closing bonds are not counted
   twice.
3. **Critical size without an assumed threshold.** Radius-of-gyration
   power-law scaling and fragmentation/survival statistics converge on the
   same critical size.
4. **Regime-specific kinetics.** Single-particle species follow first-order
   chains (Bateman/convolution), the embryo regime follows an
   aggregation-fragmentation equation and supercritical clusters follow a
   constant-kernel moment closure. The quenching/cooling stage is bridged
   with the parameter-free kinetic-theory scaling k ∝ √T.

## Installation

```bash
pip install -r requirements.txt
```

The package depends only on `numpy` and `scipy`.

## Quick start

```python
from bfs_kinetics import read_frames, determine_cutoffs, identify

# LAMMPS type numbers are arbitrary, so supply the type -> element map
# for the trajectory being analysed.
element_map = {1: "A", 2: "B", 3: "C"}
central = "A"

frames = read_frames("trajectory.dump")
cutoffs = determine_cutoffs(frames,
                            pairs=[("A", "B"), ("A", "C")],
                            element_map=element_map)
components, _ = identify(frames[0], cutoffs, element_map,
                         central_element=central)
print(components[0]["formula"])
```

The command-line pipeline turns a dump file into cutoffs, formula counts and
cluster sizes:

```bash
python run_pipeline.py --dump trajectory.dump --out results/ \
    --element-map "1=A,2=B,3=C" --central-element A --pairs "A-B,A-C"
```

## Self-test

`demo.py` runs the complete workflow on a small synthetic trajectory built
from generic elements A (central), B (bridging) and C (terminal); no external
data are required.

```bash
python demo.py
```

A final `ALL CHECKS PASSED` line means every invariant held.

## Conventions

- Distances are in angstroms, temperatures in kelvin.
- The element map is system-specific and must be supplied by the caller;
  no mapping is assumed.
- The analysis scripts are system-independent and do not contain any trained
  interatomic potential.
