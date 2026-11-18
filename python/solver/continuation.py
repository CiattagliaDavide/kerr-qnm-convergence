"""Follow a fixed Kerr overtone from chi = 0 to chi_end (M = 1).

Each spin step predicts omega, corrects it, checks continuity of the angular
pair, then accepts or halves the step. Only accepted states enter the path.
Leaver is used on Re(omega) > 0; Heun follows the inherited continuation path
near the cut. This prescription does not by itself select a physical sheet.
"""
from dataclasses import dataclass

import numpy as np

from angular import overlap, projector_distance, schwarzschild_pair, tails_ok
from solve import L_CAP, L_STEP, SAME_A, SAME_OMEGA, TAIL_TOL, Mode, evaluate, solve_at

# Spin steps: grow after acceptance, halve after rejection.
DCHI_START = 0.0025
DCHI_MAX = 0.005
GROW = 1.25
DCHI_MIN = 1e-6
MAX_MISSES = 10
DCHI_MIN_PATIENT = 1e-8   # delicate angular pair, but only the frequency prediction missed
MAX_SOLVES = 6000

OVERLAP_MIN = 0.9
PROJECTOR_DISTANCE_MAX = 0.35
C_STAR = 0.03037251127 - 2.95209813483j   # approximate ell=2/3 angular exceptional point, s=-2, m=2
C_STAR_RADIUS = 0.18
COND_TOUCHY = 4.0

RE_OMEGA_MIN = 1e-8       # edge of the Leaver chart
SEED_TOL = 1e-5           # recover the supplied Schwarzschild seed before walking


@dataclass(frozen=True)
class State:
    """One accepted point of the path."""
    chi: float
    omega: complex
    pair: object                  # angular Pair at c = chi*omega
    R: object                     # radial result (the kick reads its Nollert depth)
    method: str                   # "nollert" or "heun"
    guess_error: float = None     # |omega - guess|, used to set the next prediction tolerance
    step: float = None            # accepted step in chi
    heun_rungs: tuple = (0, 0)    # accepted Heun digit/radius levels

    A = property(lambda self: self.pair.A)
    b = property(lambda self: self.pair.b)              # right S coefficients on j = 2..l_max
    b_left = property(lambda self: self.pair.b_left)
    l_max = property(lambda self: self.pair.l_max)


def follow_overtone(n, seed, ell=3, chi_end=0.7, show=None):
    """Return (accepted path, stop reason); call show(chi, omega) after each acceptance."""
    mode = Mode(-2, ell, 2, n)

    anchor, _, _ = solve_here(mode, 0.0, seed, schwarzschild_pair(mode.s, mode.ell, mode.m), "nollert")
    solves = 1
    if anchor is None or abs(anchor.omega - seed) > SEED_TOL:
        return [], "Schwarzschild seed not recovered"
    path = [State(anchor.chi, anchor.omega, anchor.pair, anchor.R, anchor.method)]
    if show:
        show(0.0, anchor.omega)

    dchi, misses = DCHI_START, 0      # the step we plan, and the halvings in a row
    kick = None                    # initial d omega/d chi, calculated only if needed

    while path[-1].chi < chi_end:
        if solves >= MAX_SOLVES:
            return path, "solve budget used up"
        parent = path[-1]
        chi = min(parent.chi + dchi, chi_end)
        if np.nextafter(chi, np.inf) >= chi_end:      # one ulp short of the end = arrived
            chi = chi_end
        step = chi - parent.chi                       # the step we take: dchi, or less at the end

        # Predict omega and fix its tolerance before solving.
        guess = predict(path, chi)
        if len(path) == 1 and kick is not None:
            guess += step * kick
        leash = leash_radius(path, chi, guess)

        # Before approaching the cut, Heun must recover the accepted parent.
        method = parent.method
        halves_re_omega = guess.real <= max(RE_OMEGA_MIN, parent.omega.real - guess.real)
        if method == "nollert" and halves_re_omega:
            solves += 1
            if not heun_agrees(mode, parent):
                return path, "Leaver and Heun disagree at the parent"
            method = "heun"

        # Try the prediction first; use the parent's omega only after a nonfatal miss.
        root, guess_unresolved, fatal = solve_here(mode, chi, guess, parent.pair, method, parent.heun_rungs)
        solves += 1
        from_guess = root is not None
        if root is None and not fatal:
            root, _, fatal = solve_here(mode, chi, parent.omega, parent.pair, method, parent.heun_rungs)
            solves += 1
            if root is not None and guess_unresolved:     # the guess try was blind: maybe another root there
                return path, "one try found a root, the other could not be resolved"
        if fatal:
            return path, "numerics broke down, or off the Re(omega) > 0 chart"

        # Continue the same mode: frequency near its prediction, angular pair near its parent.
        is_child = only_omega_missed = patient = False
        if root is not None:
            touchy = is_touchy(parent, root, chi, guess)
            dp = projector_distance(parent.pair, root.pair) if touchy else 0.0
            close = abs(root.omega - guess) <= leash
            alike = overlap(parent.pair, root.pair) >= OVERLAP_MIN
            same_projector = dp <= PROJECTOR_DISTANCE_MAX
            is_child = close and alike and same_projector
            only_omega_missed = alike and not close and from_guess
            # Only the predictor failed: allow finer steps near a delicate angular pair.
            patient = touchy and only_omega_missed and same_projector

        # Accept, then grow the planned step.
        if is_child:
            path.append(State(chi, root.omega, root.pair, root.R, root.method,
                              guess_error=abs(root.omega - guess), step=step, heun_rungs=root.heun_rungs))
            dchi, misses = min(DCHI_MAX, dchi * GROW), 0
            if show:
                show(chi, root.omega)
            continue

        # One repair for a poor first prediction: calculate d omega/d chi at Schwarzschild.
        if len(path) == 1 and kick is None and only_omega_missed and parent.method == "nollert":
            kick = schwarzschild_kick(mode, parent, step)
            if kick is None:
                return path, "no reliable d omega/d chi at Schwarzschild"
            continue

        # Rejected trials never enter path. Try the same parent with a shorter step.
        dchi_min = DCHI_MIN_PATIENT if patient else DCHI_MIN
        if step <= dchi_min or dchi <= dchi_min:
            return path, "step too small"
        if misses >= MAX_MISSES and not patient:
            return path, "too many misses"
        dchi, misses = max(dchi_min, step * 0.5), misses + 1

    return path, "arrived"


def predict(path, chi):
    """omega at the next chi: straight line through the last two accepted points (flat at the start)."""
    last = path[-1]
    if len(path) == 1:
        return last.omega
    prev = path[-2]
    return last.omega + (chi - last.chi) * (last.omega - prev.omega) / (last.chi - prev.chi)


def leash_radius(path, chi, guess):
    """Allowed distance from the prediction, using its scale and the last three errors."""
    parent = path[-1]
    h = chi - parent.chi
    floor = omega_floor(guess)
    move = 0.5 * abs(guess - parent.omega)
    first_step = 2.0 * h if len(path) == 1 else 0.0    # flat first guess: room for |d omega/d chi| up to 2

    recent = []
    for i in range(max(1, len(path) - 3), len(path)):        # the anchor has no prediction error
        past = path[i]
        if i == 1:                                           # no step before it: plain (h / its step)^2
            ratio = (h / past.step) ** 2
        else:                                                # a line-guess error grows like h (h + previous h)
            ratio = h * (h + parent.step) / (past.step * (past.step + path[i - 1].step))
        recent.append(past.guess_error * ratio)
    # 4x: a margin, since the curvature of omega(chi) behind these errors changes along the path
    return max(floor, move, first_step, 4.0 * max(recent, default=0.0))


def omega_floor(omega):
    """Minimum prediction tolerance, also used to check the initial slope."""
    return max(1e-6, 1e-8 * abs(omega))


def heun_agrees(mode, parent):
    """Recover the Leaver parent with Heun: same omega, A and angular overlap."""
    root, _, _ = solve_here(mode, parent.chi, parent.omega, parent.pair, "heun")
    if root is None:
        return False
    same_omega = abs(root.omega - parent.omega) < SAME_OMEGA
    same_A = abs(root.pair.A - parent.A) < SAME_A
    alike = overlap(parent.pair, root.pair) >= OVERLAP_MIN       # same S(theta), as in the child test
    return same_omega and same_A and alike


def solve_here(mode, chi, guess, ref, method, heun_rungs=(0, 0)):
    """Return (root or None, unresolved, fatal) after one fixed-spin correction.
    Also resolve the angular tails needed to compare projectors between spin steps.
    """
    l_max = ref.l_max
    try:
        while True:
            with np.errstate(divide="raise", invalid="raise", over="raise"):    # inf or nan: fatal
                root = solve_at(mode, chi, guess, ref, method, l_max, may_switch=True, heun_rungs=heun_rungs)
            if not root.ok or tails_ok(root.pair):
                break
            if root.pair.l_max >= L_CAP:          # tails still loose on the biggest basis
                return None, True, False
            l_max, guess = root.pair.l_max + L_STEP, root.omega      # bigger basis, same chi
    except np.linalg.LinAlgError:                 # angular comparison unavailable
        return None, True, False
    except (ArithmeticError, ValueError, MemoryError):        # overflow, bad input or no memory
        return None, False, True
    except RuntimeError:                          # budget/exhaustion: a retryable miss, not proof of no root
        return None, False, False

    if root.method == "nollert" and root.omega.real <= RE_OMEGA_MIN:
        return None, False, True                  # Leaver landed off its chart
    resolved = root.angular_ok and root.radial_ok        # a miss in |F| alone still counts as resolved
    return (root if root.ok else None), not resolved, False


def is_touchy(parent, root, chi, guess):
    """Ill-conditioned angular pair, or predicted c-segment passing near C_STAR?"""
    c0 = parent.chi * parent.omega
    dc = chi * guess - c0
    t = 0.0 if dc == 0 else float(np.clip(((C_STAR - c0) * dc.conjugate()).real / abs(dc) ** 2, 0, 1))
    near_star = abs(C_STAR - (c0 + t * dc)) <= C_STAR_RADIUS        # c0 + t*dc: the closest point to C_STAR
    ill_conditioned = parent.pair.cond >= COND_TOUCHY or root.pair.cond >= COND_TOUCHY
    return near_star or ill_conditioned


def schwarzschild_kick(mode, anchor, step):
    """d omega/d chi = -F_chi/F_omega at chi=0, checked with two stencils and depths.
    Return None if their spread exceeds the next step's frequency tolerance.
    """
    depth = anchor.R.depth
    trusted = []                                      # F finite and S tails small, at every evaluation
    samples = []                                      # (slope, error of the anchor omega: |F / F_omega|)
    # sample 1 (small stencil, full depth) is the answer; 0 (big stencil) and 2 (half depth) measure its error
    for probe_depth, shrink in ((depth, 1.0), (depth, 0.5), (depth // 2, 0.5)):
        dw = np.finfo(float).eps ** 0.25 * shrink
        dchi = dw / max(1.0, abs(anchor.omega))       # c = chi*omega moves by ~dw, like omega

        def F(omega, chi):
            pair, R = evaluate(omega, mode, chi, anchor.pair, anchor.l_max, "nollert", mode.n, probe_depth)
            trusted.append(R.finite and pair.tail < TAIL_TOL)
            return R.value

        f0 = F(anchor.omega, 0.0)
        F_omega = (F(anchor.omega + dw, 0.0) - F(anchor.omega - dw, 0.0)) / (2 * dw)
        F_chi = (-3 * f0 + 4 * F(anchor.omega, dchi) - F(anchor.omega, 2 * dchi)) / (2 * dchi)
        if not np.isfinite(F_omega) or F_omega == 0 or not np.isfinite(F_chi):
            return None
        samples.append((-F_chi / F_omega, abs(f0 / F_omega)))

    slope = samples[1][0]
    err = max(e + step * abs(other - slope) for other, e in samples)     # error on omega one step away
    return slope if all(trusted) and err <= omega_floor(anchor.omega) else None
