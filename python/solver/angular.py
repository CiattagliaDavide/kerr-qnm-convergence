"""Spin-weighted spheroidal harmonics: the angular eigenvalue A and S(theta) at c = chi*omega.

Teukolsky's angular equation, x = cos(theta), convention exp(i m phi - i omega t), M = 1:

    d/dx[(1 - x^2) dS/dx] + [c^2 x^2 - 2 c s x + s + A - (m + s x)^2 / (1 - x^2)] S = 0,

so A(0) = ell(ell+1) - s(s+1). Expanding S on spin-weighted spherical harmonics gives a five-band
matrix. eigenpair selects a continuation of a reference pair; overlap and projector_distance
compare two pairs. tails_ok checks the basis used for that comparison, not the infinite expansion.
"""

from dataclasses import dataclass

import numpy as np
from scipy import linalg

L_MAX_START = 24         # first basis; the solver adds harmonics when needed
TAIL_TERMS = 3           # the last coefficients test whether the basis is large enough
LIKENESS_WEIGHT = 0.25   # weight of eigenvector likeness relative to the distance in A
P_MOVE_MAX = 1e-8        # tails_ok requires cond * tail <= this


@dataclass(frozen=True)
class Pair:
    """One angular eigenpair: A, and S(theta) as coefficients on j = max(|s|,|m|)..l_max.
    b and b_left are the right and left eigenvectors, with |b| = 1 and b_left^H b = 1."""

    s: int
    m: int
    l_max: int
    A: complex
    b: np.ndarray
    b_left: np.ndarray
    tail: float       # |last TAIL_TERMS coefficients of b|
    cond: float       # |b| |b_left| / |b_left^H b|: 1 at c = 0, large near an exceptional point (two pairs merge)


def schwarzschild_pair(s, ell, m, l_max=L_MAX_START):
    """At c = 0 the matrix is diagonal: S is the harmonic j = ell itself."""
    j_min = max(abs(s), abs(m))
    b = np.zeros(l_max - j_min + 1, complex)
    b[ell - j_min] = 1.0
    A = complex(ell*(ell + 1) - s*(s + 1))
    return Pair(s, m, l_max, A, b, b.copy(), tail=float(np.linalg.norm(b[-TAIL_TERMS:])), cond=1.0)


def angular_matrix(s, c, m, l_max):
    """A is an eigenvalue of j(j+1) - s(s+1) + 2 c s x - c^2 x^2 on the harmonics
    j = max(|s|,|m|)..l_max. Complex symmetric; not Hermitian when c is complex."""
    j = np.arange(max(abs(s), abs(m)), l_max + 1)
    # x moves j by at most one (Clebsch-Gordan): <j+1|x|j> = x_up, <j|x|j> = x_diag, <j-1|x|j> = x_down
    x_up = np.sqrt(((j + 1)**2 - m*m) / ((2*j + 3)*(2*j + 1))) * np.sqrt(((j + 1)**2 - s*s) / (j + 1)**2)
    x_diag = -m*s / (j*(j + 1))
    x_down = np.r_[0.0, x_up[:-1]]
    # the last row keeps x_up**2: we project x^2, we do not square the truncated x
    main = j*(j + 1) - s*(s + 1) + 2.0*c*s*x_diag - c*c*(x_up**2 + x_down**2 + x_diag**2)
    near = 2.0*c*s*x_up[:-1] - c*c*x_up[:-1]*(x_diag[:-1] + x_diag[1:])    # j <-> j+1
    far = -c*c*x_up[:-2]*x_up[1:-1]                                         # j <-> j+2
    return np.diag(main) + np.diag(near, 1) + np.diag(near, -1) + np.diag(far, 2) + np.diag(far, -2)


def eigenpair(s, m, c, ref, l_max):
    """Select using distance in A and eigenvector likeness to ref."""
    matrix = angular_matrix(s, c, m, l_max)
    A_all, left_all, b_all = linalg.eig(matrix, left=True, right=True, check_finite=False)

    # Candidate eigenvectors are columns. Compare on the common basis, independently of their scaling.
    common = min(ref.b.size, A_all.size)
    forward = ref.b_left[:common].conj() @ b_all[:common]
    backward = left_all[:common].conj().T @ ref.b[:common]
    likeness = np.sqrt(abs(forward * backward / np.sum(left_all.conj() * b_all, axis=0)))
    distance = abs(A_all - ref.A) / max(1.0, abs(ref.A))
    score = distance + LIKENESS_WEIGHT*(1 - np.minimum(1.0, likeness))
    best = int(np.nanargmin(score))       # exclude undefined scores at a self-orthogonal pair

    # Fix the right-vector norm and phase, then normalize the left/right pairing to one.
    b = b_all[:, best] / np.linalg.norm(b_all[:, best])
    b = b * np.exp(-1j*np.angle(b[np.argmax(abs(b))]))      # largest coefficient real and positive
    left = left_all[:, best]
    pairing = np.vdot(left, b)
    cond = float(np.linalg.norm(left) * np.linalg.norm(b) / abs(pairing))
    return Pair(s, m, l_max, complex(A_all[best]), b, left / np.conj(pairing),
                tail=float(np.linalg.norm(b[-TAIL_TERMS:])), cond=cond)


def overlap(p, q):
    """|<S_p|S_q>|: 1 for the same S. Both b have norm 1; the shorter b is zero beyond its l_max."""
    common = min(p.b.size, q.b.size)
    return float(abs(np.vdot(p.b[:common], q.b[:common])))


def projector_distance(p, q):
    """||P-Q||/(||P||+||Q||), Frobenius norm, with P = b b_left^H; zero-pad the smaller basis.

    The matrix is complex symmetric: P = b b^T/(b^T b), the bilinear spectral projector.
    """
    size = max(p.b.size, q.b.size)
    P = np.pad(np.outer(p.b, p.b_left.conj()), (0, size - p.b.size))
    Q = np.pad(np.outer(q.b, q.b_left.conj()), (0, size - q.b.size))
    return float(np.linalg.norm(P - Q) / (np.linalg.norm(P) + np.linalg.norm(Q)))


def tails_ok(pair):
    """Is the basis big enough for the projector test? If not, the caller grows it and solves again."""
    left_tail = np.linalg.norm(pair.b_left[-TAIL_TERMS:]) / np.linalg.norm(pair.b_left)
    return bool(pair.cond * max(pair.tail, left_tail) <= P_MOVE_MAX)     # P moves by about cond * tail
