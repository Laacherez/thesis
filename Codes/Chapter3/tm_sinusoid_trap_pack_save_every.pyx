# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True
import numpy as np
cimport numpy as np
from scipy.interpolate import interp1d
from libc.math cimport sin, cos, exp, pi
ctypedef np.float64_t dtype_t


# === Potential Definitions ===

cdef dtype_t phi_prime_1(dtype_t q, dtype_t alpha, dtype_t l, dtype_t pi) nogil:
    return 2 * pi * alpha / l * cos(2 * pi * q / l)

cdef dtype_t phi_prime_2(dtype_t q, dtype_t kappa) nogil:
    return q / kappa

cdef dtype_t phi_1(dtype_t q, dtype_t alpha, dtype_t l, dtype_t pi) nogil:
    return alpha * sin(2 * pi * q / l)

cdef dtype_t phi_2(dtype_t q, dtype_t kappa) nogil:
    return q**2 / (2 * kappa)


# === Equilibrium Distribution Projections ===

def _Peq_projection1D_q1(q1, alpha, l, pi):
    return exp(-phi_1(q1, alpha, l, pi))

def _Peq_projection1D_q2(q2, kappa):
    return exp(-phi_2(q2, kappa))

def Peq_projection1D_q1(q1, alpha, l, pi):
    P = np.array([_Peq_projection1D_q1(qi, alpha, l, pi) for qi in q1])
    return P / np.trapz(P, q1)

def Peq_projection1D_q2(q2, kappa):
    P = np.array([_Peq_projection1D_q2(qi, kappa) for qi in q2])
    return P / np.trapz(P, q2)


# === Inverse CDF Sampling Helpers ===

def sample_sin(np.ndarray[dtype_t, ndim=1] q_sample, Peq_projection1D_q, dtype_t alpha, dtype_t l, dtype_t pi):
    y_q = Peq_projection1D_q(q_sample, alpha, l, pi)
    cdf_y_q = np.cumsum(y_q)
    cdf_y_q = cdf_y_q / cdf_y_q.max()
    return interp1d(cdf_y_q, q_sample)

def sample_optic(np.ndarray[dtype_t, ndim=1] q_sample, Peq_projection1D_q, dtype_t kappa):
    y_q = Peq_projection1D_q(q_sample, kappa)
    cdf_y_q = np.cumsum(y_q)
    cdf_y_q = cdf_y_q / cdf_y_q.max()
    return interp1d(cdf_y_q, q_sample)


# === Friction Matrix ===

def friction_matrix(dtype_t g_11, dtype_t g_12, dtype_t g_22):
    cdef dtype_t gamma_11 = g_11**2 + g_12**2
    cdef dtype_t gamma_12 = g_12 * (g_11 + g_22)
    cdef dtype_t gamma_22 = g_22**2 + g_12**2
    return gamma_11, gamma_12, gamma_22


# === Update Step ===

cdef (dtype_t, dtype_t) update_q_values(
    dtype_t q1_current, dtype_t q2_current,
    dtype_t phi_prime_1_current, dtype_t phi_prime_2_current,
    dtype_t eta_1_current, dtype_t eta_2_current,
    dtype_t gamma_11, dtype_t gamma_12, dtype_t gamma_22,
    dtype_t gamma_ratio_11, dtype_t gamma_ratio_22,
    dtype_t tau) nogil:

    cdef dtype_t q1_next, q2_next

    q1_next = q1_current + (gamma_11 - gamma_12**2 / gamma_22)**(-1) * (
        tau * (-phi_prime_1_current + gamma_ratio_22 * phi_prime_2_current) +
        eta_1_current - gamma_ratio_22 * eta_2_current)

    q2_next = q2_current + (gamma_22 - gamma_12**2 / gamma_11)**(-1) * (
        tau * (-phi_prime_2_current + gamma_ratio_11 * phi_prime_1_current) +
        eta_2_current - gamma_ratio_11 * eta_1_current)

    return q1_next, q2_next


# === Main Simulation ===

def qi_full_arrays(
    dtype_t gamma_11, dtype_t gamma_12, dtype_t gamma_22,
    dtype_t g_11, dtype_t g_12, dtype_t g_22,
    dtype_t kappa, dtype_t alpha, dtype_t l,
    int N, dtype_t tau, int save_every):

    cdef dtype_t pi_val = 3.14159265358979323846
    cdef dtype_t gamma_ratio_22 = gamma_12 / gamma_22
    cdef dtype_t gamma_ratio_11 = gamma_12 / gamma_11

    cdef int M_cdf = 10000
    q1_theo = np.linspace(-l, l, M_cdf)
    q2_theo = np.linspace(-0.1, 0.1, M_cdf)

    inverse_cdf_q1 = sample_sin(q1_theo, Peq_projection1D_q1, alpha, l, pi_val)
    inverse_cdf_q2 = sample_optic(q2_theo, Peq_projection1D_q2, kappa)

    cdef dtype_t q1_i = inverse_cdf_q1(np.random.uniform(1e-15, 1 - 1e-15))
    cdef dtype_t q2_i = inverse_cdf_q2(np.random.uniform(1e-15, 1 - 1e-15))

    cdef int M = N // save_every
    cdef np.ndarray[dtype_t, ndim=1] q1_saved = np.empty(M, dtype=np.float64)
    cdef np.ndarray[dtype_t, ndim=1] q2_saved = np.empty(M, dtype=np.float64)

    cdef int i, j = 0
    cdef dtype_t xi1, xi2, eta1, eta2
    cdef dtype_t phi_p1, phi_p2

    for i in range(1, N):
        xi1 = np.sqrt(2 * tau) * np.random.randn()
        xi2 = np.sqrt(2 * tau) * np.random.randn()
        eta1 = g_11 * xi1 + g_12 * xi2
        eta2 = g_12 * xi1 + g_22 * xi2

        phi_p1 = phi_prime_1(q1_i, alpha, l, pi_val)
        phi_p2 = phi_prime_2(q2_i, kappa)

        q1_i, q2_i = update_q_values(q1_i, q2_i, phi_p1, phi_p2, eta1, eta2,
                                     gamma_11, gamma_12, gamma_22,
                                     gamma_ratio_11, gamma_ratio_22, tau)

        if i % save_every == 0:
            q1_saved[j] = q1_i
            q2_saved[j] = q2_i
            j += 1

    return q1_saved, q2_saved


# === Public Interface ===

def trajectory_sin_trap(int N, dtype_t g_11, dtype_t g_12, dtype_t g_22,
                        dtype_t tau, dtype_t kappa, dtype_t alpha, dtype_t l,
                        int save_every):

    gamma_11, gamma_12, gamma_22 = friction_matrix(g_11, g_12, g_22)

    return qi_full_arrays(gamma_11, gamma_12, gamma_22,
                          g_11, g_12, g_22,
                          kappa, alpha, l, N, tau, save_every)
