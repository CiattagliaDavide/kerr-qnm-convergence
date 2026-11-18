"""Correct one Kerr QNM at fixed spin, F(omega) = R(omega, A(chi*omega)) = 0.

solve_at selects the radial method. Each method solves, checks, then refines:
    angular tails or A unstable -> larger basis, restart from the corrected omega
    Nollert residual/depth      -> deeper tail, restart from the corrected omega
    Heun digits unresolved     -> more digits, restart from the original guess
    Heun radius unresolved     -> larger radius, restart from the original guess
Available refinements are tried in this order for each method. M = 1; chi = a/M.
"""
from dataclasses import dataclass, replace

import numpy as np

from angular import L_MAX_START, eigenpair
from heun import RADII, ROOT_RADIUS, SeriesTooShort, heun_residual
from radial import Radial, nollert, starting_depth

# Radial errors are divided by |F'| to estimate the error on omega.
OMEGA_TOL = 1e-10
SAME_OMEGA, SAME_A = 1e-8, 1e-7

SECANT_TOL = 1e-11
SECANT_FIRST_STEP = 1e-6
SECANT_EVALS = {"nollert": 300, "heun": 60}

TAIL_TOL = 2e-9
A_TOL = 2e-9
L_STEP, L_CAP = 8, 320
DEPTH_CAP = 320_000                  # maximum working depth; check at twice this depth
SWITCH_DEPTH_FACTOR = 8             # switch when the checked depth reaches this times the initial depth

# Work at one precision and check at the next.
HEUN_DIGITS = (45, 60, 90, 120, 150, 180, 240, 300, 360, 480, 600, 720)
START_DIGIT_TOL = 2e-8               # stability of F at the starting guess, before the secant


@dataclass(frozen=True)
class Mode:
    s: int
    ell: int
    m: int
    n: int            # overtone label; the continued fraction is inverted at n_inv (n by default)


@dataclass(frozen=True)
class Root:
    """A candidate frequency and its checks; .ok is true only when all three pass."""
    chi: float
    omega: complex
    pair: object           # angular Pair at c = chi*omega
    R: object              # radial result: Nollert at depth 2N, or Heun with more digits
    method: str            # "nollert" or "heun"
    angular_ok: bool       # S tails small, A stable when the basis grows
    radial_ok: bool        # F finite and resolved (Nollert depth, or Heun digits + radius)
    residual_ok: bool      # |F|/|F'| < OMEGA_TOL
    heun_rungs: tuple = (0, 0)   # digit/radius levels; inherited by the next step if accepted

    @property
    def ok(self):
        return self.angular_ok and self.radial_ok and self.residual_ok


def solve_at(mode, chi, guess, ref, method, l_max=L_MAX_START, n_inv=None, may_switch=False, heun_rungs=(0, 0)):
    """Correct one guess while following A from ref; return a Root (check .ok).
    Heun inherits heun_rungs. Kernel exceptions reach the caller.
    """
    n_inv = mode.n if n_inv is None else n_inv
    first_depth = starting_depth(mode.n) if method == "nollert" else None
    first_guess = guess = complex(guess)
    digit_rung, radius_rung = heun_rungs
    heun_rungs = (digit_rung, radius_rung)
    nollert_root = None

    if method == "nollert":
        root = _solve_nollert(mode, chi, guess, ref, l_max, n_inv, first_depth, heun_rungs)
        switch = root.ok and may_switch and root.R.depth >= SWITCH_DEPTH_FACTOR * first_depth
        if not switch:
            return root
        # Heun must recover this root. Precision retries still start from the original guess.
        nollert_root = root
        guess, l_max = root.omega, root.pair.l_max

    return _solve_heun(mode, chi, guess, ref, l_max, n_inv, heun_rungs,
                       first_guess=first_guess, nollert_root=nollert_root)


def _solve_nollert(mode, chi, guess, ref, l_max, n_inv, depth, heun_rungs):
    """Solve and check the continued fraction with a Nollert tail."""
    def F(omega):
        return evaluate(omega, mode, chi, ref, l_max, "nollert", n_inv, depth, None, None)[1].value

    while True:
        omega, slope = secant(F, guess, SECANT_EVALS["nollert"])
        abs_dF = abs(slope) if np.isfinite(slope) and slope != 0 else np.nan
        pair, R = evaluate(omega, mode, chi, ref, l_max, "nollert", n_inv, depth, None, None)
        angular_ok = angular_check(omega, pair, mode, chi, l_max)

        # The checked value at 2N also supplies the final residual.
        deeper = nollert(omega, chi, mode.s, mode.m, pair.A, n_inv, 2 * depth)
        R = replace(deeper, error=abs(deeper.value - R.value))
        resolved = R.error / abs_dF < OMEGA_TOL
        radial_ok = bool(R.finite and resolved)
        residual_ok = abs(R.value) / abs_dF < OMEGA_TOL
        root = Root(chi, omega, pair, R, "nollert", angular_ok, radial_ok, residual_ok, heun_rungs)
        if root.ok:
            return root

        if not angular_ok and l_max < L_CAP:
            l_max, guess = l_max + L_STEP, omega
            continue
        if not (residual_ok and resolved) and depth < DEPTH_CAP:
            depth, guess = min(2 * depth, DEPTH_CAP), omega
            continue
        return root


def _solve_heun(mode, chi, guess, ref, l_max, n_inv, heun_rungs, *, first_guess, nollert_root=None):
    """Solve and check the Heun residual, recovering nollert_root when supplied."""
    digit_rung, radius_rung = heun_rungs

    def F(omega):
        return evaluate(omega, mode, chi, ref, l_max, "heun", n_inv, None, digits, radius)[1].value

    while True:
        digits, check_digits = HEUN_DIGITS[digit_rung], HEUN_DIGITS[digit_rung + 1]
        radius, check_radii = RADII[radius_rung]
        top = check_digits == HEUN_DIGITS[-1]

        try:
            enough_digits, f_guess = heun_start_check(guess, mode, chi, ref, l_max,
                                                      digits, check_digits, radius)
            if enough_digits:
                omega, slope = secant(F, guess, SECANT_EVALS["heun"], f_guess)
        except SeriesTooShort:
            enough_digits = False
        if not enough_digits:
            if top:
                raise RuntimeError("Heun: top of the digit ladder")
            digit_rung, guess = digit_rung + 1, first_guess
            continue

        abs_dF = abs(slope) if np.isfinite(slope) and slope != 0 else np.nan
        pair, R = evaluate(omega, mode, chi, ref, l_max, "heun", n_inv, None, digits, radius)
        angular_ok = angular_check(omega, pair, mode, chi, l_max)
        R, resolved, more_digits_help = heun_check(omega, pair, R, abs_dF, mode, chi,
                                                  check_digits, radius, check_radii)
        agree = nollert_root is None or (abs(omega - nollert_root.omega) < SAME_OMEGA
                                         and abs(pair.A - nollert_root.pair.A) < SAME_A)
        radial_ok = bool(R.finite and resolved and agree)
        residual_ok = abs(R.value) / abs_dF < OMEGA_TOL
        root = Root(chi, omega, pair, R, "heun", angular_ok, radial_ok, residual_ok,
                    heun_rungs=(digit_rung, radius_rung))
        if root.ok:
            return root

        if not angular_ok and l_max < L_CAP:
            l_max, guess = l_max + L_STEP, omega
            continue
        if angular_ok and R.finite and agree:
            if more_digits_help and not top:
                digit_rung, guess = digit_rung + 1, first_guess
                continue
            if not resolved and radius_rung + 1 < len(RADII):
                radius_rung, guess = radius_rung + 1, first_guess
                continue
        return root


def angular_check(omega, pair, mode, chi, l_max):
    """Small angular tail and stable A after adding L_STEP harmonics?"""
    dA = np.nan                        # no comparison at L_CAP: unresolved
    if l_max < L_CAP:
        dA = abs(eigenpair(mode.s, mode.m, chi * omega, pair, l_max + L_STEP).A - pair.A)
    return pair.tail < TAIL_TOL and dA < A_TOL


def evaluate(omega, mode, chi, ref, l_max, method, n_inv, depth=None, digits=90, radius=ROOT_RADIUS):
    """omega -> A(chi*omega) -> F; return the angular pair and radial residual."""
    pair = eigenpair(mode.s, mode.m, chi * omega, ref, l_max)
    if method == "heun":
        value = heun_residual(omega, chi, mode.s, mode.m, pair.A, radius=radius, digits=digits)
        return pair, Radial(value, bool(np.isfinite(value)))
    return pair, nollert(omega, chi, mode.s, mode.m, pair.A, n_inv, depth)


def secant(F, guess, max_evals, f_guess=None):
    """Return the last omega and secant slope. Reuse f_guess when available.
    A small step stops iteration; the later resolution checks decide acceptance.
    """
    w0, w = guess, guess + SECANT_FIRST_STEP * max(1.0, abs(guess))
    f0 = F(w0) if f_guess is None else f_guess
    f = F(w)
    slope = np.nan
    for _ in range(max_evals - 2):                  # the two starting points count
        if not np.isfinite(f - f0) or f == f0:      # F not finite, or flat: no secant through it
            break
        slope = (f - f0) / (w - w0)
        w0, f0, w = w, f, w - f / slope
        if abs(w - w0) <= SECANT_TOL * max(1.0, abs(w)):
            break
        f = F(w)
    return w, slope


def heun_start_check(start, mode, chi, ref, l_max, digits, check_digits, radius):
    """Check F at the initial guess with more digits and the same A; return (stable, F)."""
    pair, R = evaluate(start, mode, chi, ref, l_max, "heun", n_inv=None, digits=digits, radius=radius)
    higher = heun_residual(start, chi, mode.s, mode.m, pair.A, radius=radius, digits=check_digits)
    stable = bool(R.finite and np.isfinite(higher) and abs(R.value - higher) < START_DIGIT_TOL)
    return stable, R.value


def heun_check(omega, pair, R, abs_dF, mode, chi, check_digits, radius, check_radii):
    """Compare F across digits and radii, using |F'| to estimate the error on omega.
    Return the checked residual, whether it is resolved, and whether digits need refinement.
    """
    values = [heun_residual(omega, chi, mode.s, mode.m, pair.A, radius=r, digits=check_digits)
              for r in check_radii]
    center = values[check_radii.index(radius)]       # more digits, same radius
    digit_err = abs(R.value - center)
    err = max(digit_err, max(abs(v - center) for v in values))
    checked = Radial(center, bool(np.isfinite([R.value, *values]).all()))
    if not np.isfinite(abs_dF):
        return checked, False, True            # no usable slope: unresolved, and more digits may help
    resolved = err / abs_dF < OMEGA_TOL        # same root with more digits, at every radius
    more_digits_help = not resolved and digit_err / abs_dF >= OMEGA_TOL
    return checked, resolved, more_digits_help
