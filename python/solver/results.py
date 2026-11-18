"""Save and load modes without running the solver again.

One NPZ per mode, mode_l<ell>_n<n>.npz, never overwritten (a second run of the same n gets _2, _3, ...),
plus one line per mode in modes.csv, to read by eye. To open a folder: load_modes(folder, ell).

Each state stores chi, omega (M = 1), A, the radial method and the angular coefficients b_j,
j = 2..l_max. Rows are zero padded as l_max grows. They give S and N = sum_j b_j^2, with no
conjugation. The left vector is recoverable from b because the angular matrix is complex symmetric.
"""
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARRIVED_TOL = 1e-12       # the walk lands exactly on chi_end; this only absorbs roundoff


@dataclass
class Saved:
    """One followed mode as saved: n, ell and, along the path, chi, omega, A, method, b, l_max."""
    n: int
    ell: int
    chi: np.ndarray
    omega: np.ndarray
    A: np.ndarray
    method: np.ndarray             # "nollert" or "heun" at each chi
    b: np.ndarray = None           # (len(chi), longest basis), zero padded; None if S was not saved
    l_max: np.ndarray = None       # b[k, :l_max[k] - 1] is j = 2..l_max at the k-th chi
    why: str = ""                  # "arrived", or why the walk stopped
    chi_end: float = 0.7

    @property
    def arrived(self):
        return abs(self.chi[-1] - self.chi_end) < ARRIVED_TOL


def save_mode(folder, n, ell, path, why, chi_end=0.7):
    """Save a nonempty path from follow_overtone, append its CSV summary and return the NPZ path."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    b = np.zeros((len(path), max(len(state.b) for state in path)), dtype=complex)
    for k, state in enumerate(path):
        b[k, :len(state.b)] = state.b

    filename = folder / f"mode_l{ell}_n{n:03d}.npz"
    run = 1
    while filename.exists():                            # preserve previous runs of this n
        run += 1
        filename = folder / f"mode_l{ell}_n{n:03d}_{run}.npz"
    np.savez(filename, n=n, ell=ell, why=str(why), chi_end=chi_end, b=b,
             chi=np.array([state.chi for state in path], dtype=float),
             omega=np.array([state.omega for state in path], dtype=complex),
             A=np.array([state.A for state in path], dtype=complex),
             method=np.array([state.method for state in path]), l_max=np.array([state.l_max for state in path]))

    table = folder / "modes.csv"
    first = not table.exists()
    end = path[-1]
    with open(table, "a", newline="") as f:
        out = csv.writer(f)
        if first:
            out.writerow(["n", "ell", "chi", "omega_re", "omega_im", "A_re", "A_im", "states", "why", "file"])
        out.writerow([n, ell, end.chi, end.omega.real, end.omega.imag, end.A.real, end.A.imag,
                      len(path), why, filename.name])
    return filename


def load_mode(file):
    """One NPZ written by save_mode."""
    with np.load(file) as z:
        return Saved(n=int(z["n"]), ell=int(z["ell"]), chi=z["chi"], omega=z["omega"], A=z["A"],
                     method=z["method"], b=z["b"], l_max=z["l_max"], why=str(z["why"]),
                     chi_end=float(z["chi_end"]))


def load_modes(folder, ell):
    """Every saved mode of one ell in folder, sorted by n. Repeated runs of one n all come back."""
    files = sorted(Path(folder).glob(f"mode_l{ell}_n*.npz"))        # n020 before n020_2
    return sorted((load_mode(f) for f in files), key=lambda m: m.n)
