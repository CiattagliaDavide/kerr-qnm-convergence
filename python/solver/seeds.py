"""Schwarzschild frequencies used to start the spin continuation. Units M = 1.

The supplied tables contain ell=2, n=0..1001 and ell=3, n=0..350, for s=-2, m=2.
The ell=2 table is from Berti, Cardoso & Starinets, Class. Quantum Grav. 26, 163001 (2009).
Further ell=3 seeds are predicted from three neighbours and corrected at chi=0.

The label n follows the supplied tables. For ell=3 it skips M omega=-10i between
n=39 and 40. Extending that table uses inversion index n+1; the spin walk uses n.
The inversion index changes the conditioning of the continued fraction, not its exact zeros.
"""
import csv
from pathlib import Path

from angular import schwarzschild_pair
from solve import Mode, solve_at

DATA = Path(__file__).resolve().parent.parent / "data"
SEED_MOVE_MAX = 0.05       # below the overtone spacing and mirror-root separation
_oblate = {}              # frequencies already read or solved in this session


def read_table(ell):
    """Read n -> M omega from the supplied Schwarzschild table."""
    with (DATA / f"schwarzschild_l{ell}.csv").open() as f:
        return {int(row["n"]): complex(float(row["omega_re"]), float(row["omega_im"]))
                for row in csv.DictReader(f)}


def schwarzschild_seed(ell, n):
    """Return the Schwarzschild frequency used to start overtone n."""
    if ell == 2:
        return read_table(ell)[n]
    if ell == 3:
        if not _oblate:
            _oblate.update(read_table(ell))
        extend(_oblate, n)
        return _oblate[n]
    raise ValueError(f"no seeds for ell={ell}")


def extend(seeds, n_max):
    """Predict each missing ell=3 seed, then solve and check it at chi=0."""
    for n in range(max(seeds) + 1, n_max + 1):
        guess = 3 * seeds[n - 1] - 3 * seeds[n - 2] + seeds[n - 3]
        root = solve_at(Mode(-2, 3, 2, n), 0.0, guess, schwarzschild_pair(-2, 3, 2), "nollert", n_inv=n + 1)
        close = abs(root.omega - guess) < SEED_MOVE_MAX
        if not (root.ok and close):
            raise RuntimeError(f"seed n={n} refused: omega={root.omega}, guess={guess}, ok={root.ok}")
        seeds[n] = root.omega
    return seeds
