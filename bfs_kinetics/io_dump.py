# -*- coding: utf-8 -*-
"""LAMMPS dump trajectory reader.

Maps the integer atom types used in a trajectory to element symbols. LAMMPS
type numbers are arbitrary, so the mapping is system-specific and must be
supplied by the caller; no mapping is assumed here.
"""
from __future__ import annotations
import numpy as np

# No default mapping: LAMMPS type numbers differ between systems. Pass an
# explicit {type: symbol} map to Frame.symbols and the analysis functions.
DEFAULT_ELEMENTS = {}


class Frame:
    """A single trajectory frame.

    Attributes
    ----------
    timestep : int
    box : np.ndarray, shape (3,)
        Periodic box lengths Lx, Ly, Lz.
    ids : np.ndarray[int], shape (N,)
    types : np.ndarray[int], shape (N,)
    pos : np.ndarray[float], shape (N, 3)
        Cartesian coordinates wrapped into [0, L).
    """

    __slots__ = ("timestep", "box", "ids", "types", "pos")

    def __init__(self, timestep, box, ids, types, pos):
        self.timestep = int(timestep)
        self.box = np.asarray(box, dtype=float)
        self.ids = np.asarray(ids, dtype=int)
        self.types = np.asarray(types, dtype=int)
        self.pos = np.asarray(pos, dtype=float)

    @property
    def n_atoms(self) -> int:
        return len(self.ids)

    def index_of_id(self) -> dict:
        return {int(a): i for i, a in enumerate(self.ids)}

    def symbols(self, element_map=DEFAULT_ELEMENTS):
        """Element symbol for each atom. If no map is supplied, fall back to a
        neutral placeholder ``X{type}`` so the geometry can still be analysed."""
        if not element_map:
            return np.array([f"X{int(t)}" for t in self.types])
        return np.array([element_map.get(int(t), "X") for t in self.types])


def iter_dump(path: str):
    """Yield every frame of a text LAMMPS dump file as :class:`Frame`."""
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                return
            if "ITEM: TIMESTEP" not in line:
                continue
            timestep = int(f.readline())
            line = f.readline()  # ITEM: NUMBER OF ATOMS
            n_atoms = int(f.readline())
            line = f.readline()  # ITEM: BOX BOUNDS ...
            xlo, xhi = map(float, f.readline().split())
            ylo, yhi = map(float, f.readline().split())
            zlo, zhi = map(float, f.readline().split())
            box = np.array([xhi - xlo, yhi - ylo, zhi - zlo])
            line = f.readline()  # ITEM: ATOMS ...
            headers = line.split()[2:]
            id_i = headers.index("id")
            type_i = headers.index("type")
            scaled = {"xs", "ys", "zs"}.issubset(headers)
            if scaled:
                xi, yi, zi = (headers.index(c) for c in ("xs", "ys", "zs"))
            else:
                xi, yi, zi = (headers.index(c) for c in ("x", "y", "z"))
            ncol = len(headers)
            ids = np.empty(n_atoms, dtype=int)
            types = np.empty(n_atoms, dtype=int)
            pos = np.empty((n_atoms, 3), dtype=float)
            for k in range(n_atoms):
                parts = f.readline().split()
                if len(parts) < ncol:
                    parts += f.readline().split()
                ids[k] = int(parts[id_i])
                types[k] = int(parts[type_i])
                if scaled:
                    pos[k] = (xlo + float(parts[xi]) * box[0],
                              ylo + float(parts[yi]) * box[1],
                              zlo + float(parts[zi]) * box[2])
                else:
                    pos[k] = (float(parts[xi]), float(parts[yi]), float(parts[zi]))
            yield Frame(timestep, box, ids, types, pos % box)


def read_frames(path: str, step_interval: int | None = None):
    """Read frames, optionally keeping only timesteps divisible by ``step_interval``."""
    frames = []
    for fr in iter_dump(path):
        if step_interval is None or fr.timestep % step_interval == 0:
            frames.append(fr)
    return frames


def write_dump(path: str, frames):
    """Write frames to a text LAMMPS dump using absolute coordinates (x y z)."""
    with open(path, "w") as f:
        for fr in frames:
            order = np.argsort(fr.ids)
            ids, types, pos = fr.ids[order], fr.types[order], fr.pos[order]
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{int(fr.timestep)}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{fr.n_atoms}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            for L in fr.box:
                f.write(f"0.0 {float(L):.6f}\n")
            f.write("ITEM: ATOMS id type x y z\n")
            for a, t, p in zip(ids, types, pos):
                f.write(f"{int(a)} {int(t)} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
