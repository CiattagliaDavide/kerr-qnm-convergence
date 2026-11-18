"""Leaver's radial continued fraction for Kerr QNMs, closed by a Nollert tail.

Units M = 1, chi = a/M, exp(-i omega t). A is the angular eigenvalue of angular.py, not the shifted
radial lambda. Leaver peels the horizon and infinity behaviour off the radial function and expands
the rest as sum_k a_k u^k, u = (r - r_+)/(r - r_-), with a_{-1} = 0 and

    alpha_k a_{k+1} + beta_k a_k + gamma_k a_{k-1} = 0.

The QNM condition selects the minimal solution, whose series converges at infinity (u = 1).
The two recurrence solutions behave as exp(+-2 u1 sqrt(k)), u1 = sqrt(-4i kappa omega), with
kappa = sqrt(1-chi^2). On the negative imaginary omega axis Re u1 = 0: neither exponential is
smaller. heun.py handles continuation through this cut.

To evaluate the condition, nollert advances L_k = -a_{k-1}/a_k from the horizon (L_0 = 0)
and T_k = -a_{k+1}/a_k from the large-k tail. They meet at k = n_inv:

    F(omega) = beta_n - gamma_n L_n - alpha_n T_n,    n = n_inv,       F = 0 at a QNM.

n_inv is the inversion index, usually chosen as the overtone n. The caller supplies A and omega;
this module evaluates F, while solve.py finds its zero.

Sources (CZ = Cook & Zalutskiy, PRD 90, 124021 (2014), arXiv:1410.7698v2):
    recurrence           CZ Sec. II.C, Eqs. (23)-(32)
    Nollert tail         CZ Eqs. (34), (38), (39); Nollert, PRD 47, 5253 (1993)
    inverted fraction    CZ Eq. (44); Leaver, Proc. R. Soc. A 402, 285 (1985)
u4 and u5 are a local extension checked through k^-3 in the ratio equation; see nollert_u.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Radial:
    """F at one omega and A, with what the solver needs to judge it (Heun results use it too)."""
    value: complex                 # F(omega), zero at a QNM
    finite: bool
    error: float = float("nan")    # |F(2N) - F(N)|, once the solver has compared two depths
    depth: int | None = None       # Nollert depth N (None for Heun)


def nollert(omega, chi, s, m, A, n_inv, depth):
    """Match the horizon and tail fractions at n_inv; return F and its finite-value flag."""
    D = leaver_D(omega, chi, s, m, A)
    # Build coefficients together, then use Python complex in the horizon loop.
    alpha, beta, gamma = (coeffs.tolist() for coeffs in recurrence(np.arange(depth + 1), D))

    L = 0.0j                                                    # up from the horizon: a_{-1} = 0
    for k in range(n_inv):
        L = alpha[k] / (beta[k] - gamma[k] * L)                 # L_{k+1}

    # Nollert starts the downward fraction (CZ Eqs. 34, 39). Five terms reduce its depth error:
    # at the last Leaver point of prolate n=100, |F(2N)-F(N)| is 1.1e-7 rather than 2.6e-4 with three.
    u1, u2, u3, u4, u5 = nollert_u(D)
    N = depth
    T = -(1.0 + u1 / np.sqrt(N) + u2 / N + u3 / N**1.5 + u4 / N**2 + u5 / N**2.5)
    # T and F remain numpy complex128: overflow raises under the caller's np.errstate(raise).
    for k in range(depth, n_inv, -1):
        T = gamma[k] / (beta[k] - alpha[k] * T)                 # T_{k-1}

    F = beta[n_inv] - gamma[n_inv] * L - alpha[n_inv] * T       # the recurrence at k = n_inv, over a_{n_inv}
    return Radial(complex(F), bool(np.isfinite(F)), depth=depth)


def leaver_D(omega, chi, s, m, A):
    """The five D's of CZ Eq. (31): the radial ODE once horizon and infinity behaviour are peeled off."""
    kappa = np.sqrt(1.0 - chi * chi)                 # sqrt(1 - chi^2) = (r_plus - r_minus)/2, as in heun.py
    r_plus = 1.0 + kappa
    r_minus = 1.0 - kappa
    sigma_plus = (2.0 * omega * r_plus - m * chi) / (2.0 * kappa)
    sigma_minus = (2.0 * omega * r_minus - m * chi) / (2.0 * kappa)

    # ingoing at the horizon: zeta_+, xi_-, eta_+ of CZ Eq. (28). Infinity is left to the fraction.
    zeta = 1j * omega
    xi = -s - 1j * sigma_plus
    eta = -1j * sigma_minus

    # confluent-Heun parameters of CZ Eq. (27) (the _h is there so they are not
    # mistaken for the recurrence alpha_k, gamma_k below)
    p = kappa * zeta
    alpha_h = 1.0 + 2.0 * s + xi + eta - 2.0 * zeta
    gamma_h = 1.0 + s + 2.0 * eta
    delta_h = 1.0 + s + 2.0 * xi
    sigma_h = (
        A
        + chi * chi * omega * omega
        - 8.0 * omega * omega
        + p * (2.0 * alpha_h + gamma_h - delta_h)
        + (1.0 + s - 0.5 * (gamma_h + delta_h)) * (s + 0.5 * (gamma_h + delta_h))
    )

    D = np.empty(5, dtype=np.complex128)
    D[0] = delta_h
    D[1] = 4.0 * p - 2.0 * alpha_h + gamma_h - delta_h - 2.0
    D[2] = 2.0 * alpha_h - gamma_h + 2.0
    D[3] = alpha_h * (4.0 * p - delta_h) - sigma_h
    D[4] = alpha_h * (alpha_h - gamma_h + 1.0)
    return D


def recurrence(k, D):
    """alpha_k, beta_k, gamma_k of the recurrence above (CZ Eq. 32). k (an int or an array of ints)
    is the index of the series coefficient a_k, not the overtone."""
    alpha = k * k + (D[0] + 1.0) * k + D[0]
    beta = -2.0 * k * k + (D[1] + 2.0) * k + D[3]
    gamma = k * k + (D[2] - 3.0) * k + D[4] - D[2] + 2.0
    return alpha, beta, gamma


def nollert_u(D):
    """u1..u5 in a_{k+1}/a_k = 1 + sum_j u_j k^(-j/2) at large k.

    Each u_j solves [x^(j+1)] of the ratio equation, 2 u1 u_j + rest_j = 0, with x = k^(-1/2).
    u4 and u5 follow from the same ratio equation, expanded through k^-3.
    """
    D0, D1, D2, D3, D4 = D
    # a_k ~ exp(2 u1 sqrt(k)) and we want the tail that dies, Re u1 < 0. np.sqrt gives Re >= 0,
    # so flip; exactly on the axis (Re u1 = 0) keep Im u1 <= 0, just a convention.
    # D0 + D1 + D2 = 4i kappa omega, so this flip is the cut on the negative imaginary omega axis
    u1 = np.sqrt(-(D0 + D1 + D2))
    if u1.real > 0.0 or u1.imag > 0.0:
        u1 = -u1
    u2 = -0.5 * D0 + 0.5 * D2 + 0.5 * u1 * u1 - 1.75    # CZ Eq. (38b)

    rest3 = (                                            # CZ Eq. (38c)
        D0 * u2 + D0 + D2 * u1 * u1 - D2 * u2 - D2 + D3 + D4
        + u1**4 - 3.0 * u1 * u1 * u2 - 2.0 * u1 * u1 + u2 * u2 + 3.0 * u2 + 2.0
    )
    u3 = -rest3 / (2.0 * u1)

    rest4 = (                                            # coefficient equation for u4
        D0 * u1 + D0 * u3 - D2 * u1**3 + 2.0 * D2 * u1 * u2 + 0.5 * D2 * u1 - D2 * u3 - D4 * u1
        - u1**5 + 4.0 * u1**3 * u2 + 1.5 * u1**3 - 3.0 * u1 * u1 * u3 - 3.0 * u1 * u2 * u2
        - 3.0 * u1 * u2 - 0.875 * u1 + 2.0 * u2 * u3 + 2.5 * u3
    )
    u4 = -rest4 / (2.0 * u1)

    rest5 = (                                            # coefficient equation for u5
        D0 * u2 + D0 * u4 + D2 * u1**4 - 3.0 * D2 * u1 * u1 * u2 + 2.0 * D2 * u1 * u3
        + D2 * u2 * u2 - D2 * u4 + D4 * u1 * u1 - D4 * u2
        + u1**6 - 5.0 * u1**4 * u2 - u1**4 + 4.0 * u1**3 * u3 + 6.0 * u1 * u1 * u2 * u2
        + 3.0 * u1 * u1 * u2 - 3.0 * u1 * u1 * u4 - 6.0 * u1 * u2 * u3 - 2.0 * u1 * u3
        - u2**3 - u2 * u2 + 2.0 * u2 * u4 + u3 * u3 + 2.0 * u4
    )
    u5 = -rest5 / (2.0 * u1)
    return u1, u2, u3, u4, u5


def starting_depth(n):
    """Nollert depth N to start with for overtone n. A work budget that grows with n, not physics.
    From n = 799 on it starts at or above the solver's doubling cap (320000), so it never doubles."""
    return min(640_000, max(4_000, 400 * (n + 1)))
