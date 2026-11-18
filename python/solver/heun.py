"""Kerr radial residual from the confluent Heun equation, used near and past Re(omega) = 0.

Leaver's tail changes across the cut of u1 = sqrt(-4i kappa omega), on the negative imaginary omega
axis (radial.py). Here we integrate from the horizon along a ray with RAY_TILT < 0 to continue
through that region. This specifies the path, not which Green-function sheet is physical.

Units M = 1, chi = a/M, exp(-i omega t). Inside we use r_hat = r/(2M), a = chi/2, w = 2 omega,
kappa = sqrt(1 - chi^2) = r_hat_plus - r_hat_minus, sigma_pm = (w r_hat_pm - a m)/kappa, and
x = -(r_hat - r_hat_plus)/kappa: the horizon is x = 0, the inner horizon x = 1, infinity |x| -> infinity.

    R = (r_hat-r_hat_plus)^(-s-i*sigma_plus) (r_hat-r_hat_minus)^(-s+i*sigma_minus) exp(-i*w*r_hat) * H(x),
    H'' + (gamma/x + delta/(x-1) + epsilon) H' + (alpha*x - q)/(x*(x-1)) H = 0,     H(0) = 1.

H(0) = 1 and the prefactor select the ingoing horizon wave. At large |x| (alpha/epsilon = 1-2s),

    x^(1-2s) H = A_in (1 + O(1/x))  +  A_out exp(2i omega (r - r_plus)) x^(4i omega - 2s) (1 + O(1/x))
                 ingoing from infinity        outgoing

and a QNM has A_in = 0. heun_residual starts with a horizon series and advances H, H' by Taylor
steps to |r-r_plus| = 2M * radius. It returns x^(1-2s) H, a finite-radius estimate of A_in.
The ray nearly follows the steepest decay of the outgoing exponential: its modulus at the endpoint
is exp[-4|omega| radius cos(pi RAY_TILT/2)]. The power x^(4i omega-2s) also affects the outgoing term.

The incoming O(1/x) correction multiplies A_in and preserves its zero if that factor is nonzero.
The outgoing remainder shifts the finite-radius zero, so solve.py checks radius and digit changes.
The ray turns with arg(omega): this finite-radius residual is not exactly analytic in omega,
although A_in is locally analytic on its sheet.
"""
import math

import mpmath as mp

# (root radius, check radii), with radius = |r-r_plus|/(2M). solve.py starts on the parent's pair
# and enlarges it if needed. Repeating the root radius with more digits also checks precision.
RADII = ((3.0, (2.5, 3.0, 3.5)),     # (root radius, check radii): fewer working digits, enough up to n ~ 200
         (5.0, (4.0, 5.0, 6.0)))     # smaller outgoing remainder, for when the root moves with the radius
ROOT_RADIUS, CHECK_RADII = RADII[-1]  # defaults for a single call

RAY_TILT = -0.08         # -7.2 degrees from steepest decay; passes x=1 on the same side while
                        # Re(omega)/|Im(omega)| > -tan(7.2 deg) = -0.126 (saved walks: > -0.006)
HORIZON_STEP = 0.2        # the horizon series is summed out to |x| = 0.2; it converges up to |x| = 1
STEP_FRACTION = 0.45      # a Taylor step covers this much of the way to the nearest singular point (x = 0, 1)
GUARD_DIGITS = 45         # working digits kept beyond the fall of the outgoing wave (see heun_residual)
TOL_GUARD_DIGITS = 30     # and at least this many beyond the series tolerance 10^-digits
MAX_WORK_DIGITS = 4000    # a cost guard, not physics: at radius 6 it allows |Im omega| up to ~380
MAX_TERMS = 3000          # longest local series: the top rung (720 digits) needs ~2100, as 0.45^2100 ~ 1e-730
MAX_TRIES = 20000         # Taylor steps tried along a ray, halved ones included; a normal ray takes ~20


class SeriesTooShort(RuntimeError):
    """A local series needs more than MAX_TERMS terms. Along the ray the step is halved; if it is the
    horizon series, the error reaches the solver, which climbs its digit ladder."""


def heun_residual(omega, chi, s, m, A, radius=ROOT_RADIUS, digits=90):
    """Return the finite-radius estimate x^(1-2s) H of A_in; digits sets the series tolerance 10^-digits."""
    # Near a QNM the outgoing wave is small, but rounding can feed the incoming one.
    # Carry its exponential decay in extra digits (|Im omega| ~ |omega| in this regime).
    lost_digits = math.ceil(4*abs(omega.imag)*radius/math.log(10))
    work_digits = max(digits + TOL_GUARD_DIGITS, lost_digits + GUARD_DIGITS)
    if work_digits > MAX_WORK_DIGITS:
        raise RuntimeError("Heun working-precision budget exhausted")

    with mp.workdps(work_digits):
        p, kappa = parameters(omega, chi, s, m, A)
        tol = mp.power(10, -digits)

        # x points opposite to r-r_plus. This ray makes the outgoing exponential decay.
        ray_angle = mp.pi*(1 + mp.mpf(RAY_TILT))/2 - mp.arg(mp.mpc(omega))
        target = -mp.mpf(radius)*mp.exp(1j*ray_angle)/kappa
        x = mp.mpf(HORIZON_STEP)*target/abs(target)
        H, dH = from_horizon(p, x, tol)

        steps_tried = 1              # count the horizon series too
        while x != target:
            remaining = target - x
            max_step = mp.mpf(STEP_FRACTION)*min(abs(x), abs(x - 1))     # nearest singularity
            last_step = abs(remaining) <= max_step
            step = remaining if last_step else max_step*remaining/abs(remaining)
            while True:
                if steps_tried >= MAX_TRIES:
                    raise RuntimeError("Heun Taylor-path budget exhausted")
                steps_tried += 1
                try:
                    H, dH = taylor_step(p, x, H, dH, step, tol)
                    break
                except SeriesTooShort:
                    step /= 2        # same centre, half the step: H, dH and x have not moved
                    last_step = False
            x = target if last_step else x + step      # land exactly on target on the final step

        return complex(H*target**(1 - 2*s))      # x^(alpha/epsilon) H, alpha/epsilon = 1 - 2s exactly


def parameters(omega, chi, s, m, A):
    """(alpha, gamma, delta, epsilon, q) of the Heun equation above, and kappa = sqrt(1 - chi^2)."""
    a, w, A = mp.mpf(chi)/2, 2*mp.mpc(omega), mp.mpc(A)
    kappa = mp.sqrt(1 - 4*a*a)
    # in the usual variable z = 1 - x = (r_hat - r_hat_minus)/kappa: inner horizon at z = 0, horizon at z = 1
    alpha_z = 2j*kappa*(2*s - 1)*w
    gamma_z = 1 - s - 1j*(2*a*m - w)/kappa - 1j*w
    delta_z = 1 - s + 1j*(2*a*m - w)/kappa - 1j*w
    epsilon_z = -2j*kappa*w
    q_z = A + w*(1j - 1j*kappa - 2*a*m + a*a*w) + s*(2 + 2j*(kappa - 1)*w)
    # back to x: swap gamma and delta, (alpha, epsilon, q) -> (-alpha, -epsilon, q - alpha)
    return (-alpha_z, delta_z, gamma_z, -epsilon_z, q_z - alpha_z), kappa


def sum_series(coefficients, step, value, slope, start, tol):
    """Add sum_{j >= start} c_j step^j to value, and its derivative to slope (H and H' go together).

    Stop when four terms in a row are small in BOTH sums, and never before order 14:
    one coefficient passing close to zero must not stop the sum."""
    power = step**(start - 1)
    small_in_a_row = 0
    for j, c in enumerate(coefficients, start):
        slope_term = j*c*power
        power *= step
        term = c*power
        value += term
        slope += slope_term
        small = (abs(term) < tol*max(1, abs(value))           # relative, or absolute once the sum is below 1
                 and abs(slope_term) < tol*max(1, abs(slope)))
        small_in_a_row = small_in_a_row + 1 if j >= 14 and small else 0
        if small_in_a_row >= 4:
            return value, slope
    raise SeriesTooShort()


def from_horizon(p, x, tol):
    """H, H' at x from the Frobenius series at the horizon, H = sum c_j x^j with c_0 = 1.

    Exponent 0, not 1 - gamma: with the prefactor of R, this is the wave falling into the horizon."""
    alpha, gamma, delta, epsilon, q = p

    def coefficients():
        c_prev, c = mp.mpc(0), mp.mpc(1)       # c_{-1}, c_0
        for j in range(MAX_TERMS - 1):
            c_next = ((j*(j - 1) + (gamma + delta - epsilon)*j - q)*c
                      + (epsilon*(j - 1) + alpha)*c_prev)/((j + 1)*(j + gamma))
            yield c_next
            c_prev, c = c, c_next

    return sum_series(coefficients(), x, mp.mpc(1), mp.mpc(0), 1, tol)


def taylor_step(p, x0, H, dH, step, tol):
    """H, H' at x0 + step from the Taylor series around x0, H = sum c_j t^j with t = x - x0."""
    alpha, gamma, delta, epsilon, q = p
    # times x(x-1), in t: (A0 + A1 t + t^2) H'' + (B0 + B1 t + epsilon t^2) H' + (C0 + alpha t) H = 0
    A0, A1 = x0*(x0 - 1), 2*x0 - 1
    B0 = epsilon*x0**2 + (gamma + delta - epsilon)*x0 - gamma
    B1 = 2*epsilon*x0 + gamma + delta - epsilon
    C0 = alpha*x0 - q

    def coefficients():
        c_prev, c, c_next = mp.mpc(0), H, dH       # c_{j-1}, c_j, c_{j+1}; c_0 = H, c_1 = H'
        for j in range(MAX_TERMS - 2):
            c_new = -((A1*j*(j + 1) + B0*(j + 1))*c_next
                      + (j*(j - 1) + B1*j + C0)*c
                      + (epsilon*(j - 1) + alpha)*c_prev)/(A0*(j + 2)*(j + 1))
            yield c_new
            c_prev, c, c_next = c, c_next, c_new

    return sum_series(coefficients(), step, H + dH*step, dH, 2, tol)
