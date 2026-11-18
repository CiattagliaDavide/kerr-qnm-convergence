"""Leading high-damping references, for the guide lines of the figures. M = 1, s = -2.

The solver never reads these: they are drawn next to the computed modes and compared with them.

    horizon(chi)      r_plus, Omega_H and kappa_H. Oblate side (Keshet & Ben-Meir, arXiv:1207.2460):
                      Re(omega) -> m Omega_H and -Im(omega) grows by kappa_H per overtone.
    keshet_hod(chi)   prolate side (Keshet & Hod, PRD 76, 061501 (2007), arXiv:0705.1179, Eqs. 13-14, Q = 0):
                      Re(omega) -> C_KH = -m delta_m / delta_0, and -Im(omega) grows by
                      S_KH = 2 pi / delta_0 per overtone.
    prolate_A(c)      large-|c| angular eigenvalue on the prolate side (Vickers & Cook, arXiv:2208.06259,
                      Eqs. 21-22): A = q K + m^2 + s(s-1) - (q^2+5)/8 + [-q(q^2+11-32m^2)/64 + q s^2 + 2i m s^2]/K,
                      K = i c, q = 2L + 1; for s = -2, m = 2, L = 0: A = K + 37/4 + (93/16 + 16i)/K.
    oblate_A(c)       the same on the oblate side (Berti, Cardoso & Casals, PRD 73, 024013 (2006), Eq. 18, to O(1)):
                      A = -c'^2 + 2 q c' - (q^2 - m^2 + 2s + 1)/2 with c' = -c in this code's convention and
                      q = 1 for (ell, m, s) = (3, 2, -2) (their Eq. 20b): A = -c^2 - 2c + 3. The comparison retains terms through c^0; higher inverse powers are omitted.

delta_0 and delta_m are action integrals of the potentials V_0 = sqrt(q0)/Delta and
V_m = -2 a r / (Delta sqrt(q0)), q0 = r^4 + a^2 r^2 + 2 a^2 r, along a closed contour around the two
complex turning points r1, r2 = conj(r1) of q0: delta_j = 2i oint V_j dr. We integrate on the lower
half, from r1 down and out to the real axis beyond r_plus, and use the conjugate symmetry for the rest.
"""
import numpy as np
from scipy.integrate import quad_vec

QUAD_TOL = 2e-11          # absolute and relative tolerance of each contour piece
MARGIN = 0.5              # the contour meets the real axis at r_plus + MARGIN
CLEARANCE = 1.0           # how far below r1 the contour runs


def horizon(chi):
    """r_plus, Omega_H, kappa_H for M = 1."""
    root = np.sqrt(1.0 - chi * chi)
    r_plus = 1.0 + root
    return r_plus, chi / (2.0 * r_plus), root / (2.0 * r_plus)


def keshet_hod(chi, m=2):
    """(C_KH, S_KH): the prolate high-damping limit of Re(omega) and the spacing of -Im(omega). 0 < chi < 1."""
    a = float(chi)
    # turning points: q0 = r (r + u) (r - r1) (r - r2), with u real and r1 = u/2 - i b
    u = 2 * a / np.sqrt(3) * np.sinh(np.arcsinh(3 * np.sqrt(3) / a) / 3)
    b = np.sqrt(a * a + 3 * u * u / 4)
    r1 = u / 2 - 1j * b
    r2 = r1.conjugate()
    r_plus = 1 + np.sqrt(1 - a * a)
    r_minus = a * a / r_plus

    def Delta(r):
        return (r - r_plus) * (r - r_minus)

    def rest(r):                                    # sqrt(q0) / sqrt(r - r1)
        return np.sqrt(r) * np.sqrt(r + u) * np.sqrt(r - r2)

    # first piece, r = r1 - i h t^2: the square-root endpoint at r1 becomes a factor t, which cancels
    h = CLEARANCE
    hroot = np.sqrt(-1j * h)

    def first_piece(t):
        r = r1 - 1j * h * t * t
        dr = -2j * h * t
        sqrt_q0 = rest(r) * hroot * t
        dr_over_sqrt_q0 = 2 * hroot / rest(r)
        return np.array([sqrt_q0 / Delta(r) * dr, -2 * a * r / Delta(r) * dr_over_sqrt_q0], dtype=complex)

    I, _ = quad_vec(first_piece, 0.0, 1.0, epsabs=QUAD_TOL, epsrel=QUAD_TOL, limit=1000)

    # then straight segments, staying below the turning point, out to the real axis beyond r_plus
    x_clear = r_plus + MARGIN
    vertices = [r1 - 1j * h, x_clear - 1j * (b + h), complex(x_clear)]
    for p, q in zip(vertices[:-1], vertices[1:]):
        dr = q - p

        def segment(t, p=p, dr=dr):
            r = p + dr * t
            sqrt_q0 = rest(r) * np.sqrt(r - r1)
            return np.array([sqrt_q0 / Delta(r), -2 * a * r / (Delta(r) * sqrt_q0)], dtype=complex) * dr

        part, _ = quad_vec(segment, 0.0, 1.0, epsabs=QUAD_TOL, epsrel=QUAD_TOL, limit=1000)
        I += part

    delta_0, delta_m = -4 * I.imag                  # 2i (I - conj I): the conjugate half of the contour
    if delta_0 < 0:                                 # one orientation for all chi: delta_0 > 0
        delta_0, delta_m = -delta_0, -delta_m
    return -m * delta_m / delta_0, 2 * np.pi / delta_0


def prolate_A(c, s=-2, m=2, L=0):
    """Vickers-Cook large-|c| expansion of A on the prolate side, through 1/(ic)."""
    K, q = 1j * c, 2 * L + 1
    return q * K + m * m + s * (s - 1) - (q * q + 5) / 8 + (-q * (q * q + 11 - 32 * m * m) / 64 + q * s * s + 2j * m * s * s) / K


def oblate_A(c, s=-2, m=2, q=1):
    """Berti-Cardoso-Casals large-|c| expansion of A on the oblate side, through O(1), with c' = -c."""
    cp = -c
    return -cp * cp + 2 * q * cp - (q * q - m * m + 2 * s + 1) / 2
