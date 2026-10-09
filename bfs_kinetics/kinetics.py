# -*- coding: utf-8 -*-
"""Kinetic models for the size regimes and the non-isothermal extension.

Single central-atom species follow first-order chains (Bateman / convolution);
the embryo regime follows a Smoluchowski aggregation-fragmentation equation
with a
monomer-addition kernel, a constant coalescence kernel and a size-dependent
evaporation term; supercritical clusters are described by a constant-kernel
moment closure for which 1/M0 is linear in time. Quenching is bridged with the
parameter-free kinetic-theory scaling k proportional to sqrt(T). Classical
nucleation quantities are included for the independent check.
"""
from __future__ import annotations
import numpy as np

KB_EV = 8.617333262145e-5  # eV/K


# ---------------------------------------------------------------- Bateman chain
def bateman(t, rates, n0=1.0):
    """Closed-form populations of an irreversible linear chain.

    ``rates`` are the first-order constants k0..k_{m-1} of steps 0->1 ...
    m-1->m; species m is the final product. Returns an array shaped
    (m+1, len(t)).
    """
    t = np.asarray(t, dtype=float)
    k = list(rates) + [0.0]  # final species does not decay
    m = len(k) - 1
    out = np.zeros((m + 1, len(t)))
    for j in range(m + 1):
        kj = k[: j + 1]
        total = np.zeros_like(t)
        for q in range(j + 1):
            denom = 1.0
            for r in range(j + 1):
                if r != q:
                    denom *= kj[r] - kj[q]
            total += np.exp(-kj[q] * t) / denom
        prefactor = np.prod(k[:j]) if j > 0 else 1.0
        out[j] = n0 * prefactor * total
    return out


def convolve_first_order(t, source, k_b):
    """Population of a first-order product fed by an arbitrary source.

    N_B(t) = integral_0^t k_b N_A(tau) exp[-k_b (t-tau)] dtau, evaluated on the
    uniform grid ``t``.
    """
    t = np.asarray(t, dtype=float)
    dt = float(t[1] - t[0]) if len(t) > 1 else 1.0
    s = np.arange(len(t)) * dt
    kernel = k_b * np.exp(-k_b * s) * dt
    return np.convolve(np.asarray(source, dtype=float), kernel, mode="full")[: len(t)]


# ------------------------------------------- Smoluchowski aggregation-fragmentation
def _kernel(a, b, k_add, alpha, k_coal):
    if min(a, b) == 1:
        return k_add * max(a, b) ** alpha
    return k_coal


def aggregation_fragmentation(t, c_initial, k_add, k_coal, k_evap,
                              alpha=0.5, beta=1.0, dilution=0.0):
    """Integrate the size-resolved aggregation-fragmentation equation.

    State vector gives number densities per size n = 1..Nmax. Monomer addition
    uses a collision cross-section power kernel, cluster coalescence a constant
    kernel, and evaporation a Becker-Doring term proportional to n^beta.
    ``dilution`` is the constant rate -(1/V)dV/dt for an expanding NPT box.
    """
    from scipy.integrate import solve_ivp

    nmax = len(c_initial)
    c0 = np.asarray(c_initial, dtype=float)

    def rhs(_, c):
        d = np.zeros(nmax)
        for n in range(1, nmax + 1):
            i = n - 1
            birth = 0.0
            for a in range(1, n):
                birth += 0.5 * _kernel(a, n - a, k_add, alpha, k_coal) \
                    * c[a - 1] * c[n - a - 1]
            death = c[i] * sum(_kernel(n, a, k_add, alpha, k_coal) * c[a - 1]
                               for a in range(1, nmax + 1))
            birth_ev = k_evap * (n + 1) ** beta * c[n] if n + 1 <= nmax else 0.0
            death_ev = k_evap * n ** beta * c[i]
            d[i] = birth - death + birth_ev - death_ev - dilution * c[i]
        # every evaporation event releases one monomer
        d[0] += sum(k_evap * m ** beta * c[m - 1] for m in range(2, nmax + 1))
        return d

    sol = solve_ivp(rhs, (float(t[0]), float(t[-1])), c0,
                    t_eval=np.asarray(t, dtype=float), rtol=1e-7, atol=1e-12)
    return sol.t, sol.y


# ------------------------------------------------- constant-kernel moment closure
def moment_closure(t, m_total, volume, k_coal, m0_initial):
    """Supercritical constant-kernel closure.

    dM0/dt = -(k_coal/(2 V)) M0^2, hence 1/M0 is linear and the mean cluster
    size is m_total/M0. Returns M0(t) and the mean size.
    """
    t = np.asarray(t, dtype=float)
    slope = k_coal / (2.0 * volume)
    m0 = m0_initial / (1.0 + slope * m0_initial * t)
    return m0, m_total / m0


def kernel_from_1_over_m0(time, m0, volume):
    """Fit the slope of 1/M0 (least squares) and return k_coal and R^2."""
    y = 1.0 / np.asarray(m0, dtype=float)
    x = np.asarray(time, dtype=float)
    a, b = np.polyfit(x, y, 1)
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return 2.0 * volume * a, r2, a, b


# ------------------------------------------------------------- non-isothermal scale
def nonisothermal_rate(temperature, t_ref, k_ref):
    """Kinetic-theory scaling k(T) = k(T_ref) sqrt(T/T_ref)."""
    return k_ref * np.sqrt(np.asarray(temperature, dtype=float) / float(t_ref))


# ------------------------------------------------------------- classical nucleation
def critical_radius(gamma, v0, temperature, ln_s):
    """r* = 2 gamma v0 / (kT ln S)."""
    kt = KB_EV * temperature
    return 2.0 * gamma * v0 / (kt * ln_s)


def critical_size(r_star, v0):
    """i* = 4 pi r*^3 / (3 v0)."""
    return 4.0 * np.pi * r_star ** 3 / (3.0 * v0)


def nucleation_barrier(i_star, ln_s):
    """Delta G*/kT = i* ln S / 2."""
    return 0.5 * i_star * ln_s


def observed_nucleation_rate(volume, t_first):
    """J_obs = 1/(V t_first), volume in cm^3 and time in seconds."""
    return 1.0 / (volume * t_first)
