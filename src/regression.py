import numpy as np
from numpy.linalg import LinAlgError
from src.utils import Value


def tau_from_parabola(a2, a2_err, a1, a1_err, a0, a0_err):
    # calculate the width of the lorentzian from the parabola coefficients
    tau = np.sqrt(np.abs((a1 ** 2 - 4 * a2 * a0) / 2)) / abs(a2)
    tau_error = (1 / tau) * (
            ((a1 ** 2 + 4 * a2 * a0) / 4 * a2 ** 3) * a2_err + (a1 / a2 ** 2) * a1_err - (
            2 / a2) * a0_err)
    return tau, tau_error


def umin_from_parabola(m_star, m_min):
    mu = np.power(10, np.abs(m_star.value - m_min) / 2.5)
    mu_error = (np.abs(m_star.value - m_min) / 2.5) * mu
    umin = np.sqrt(2 * np.sqrt(mu ** 2 / (mu ** 2 - 1)) - 2)
    umin_error = (4 * mu_error / umin) * ((mu ** 2 + mu - 1) / np.power(mu ** 2 - 1, 3 / 2))
    return umin, umin_error


def get_coefficient_errors(residuals, cut_data_time, degree):
    std_errors = [np.nan, np.nan, np.nan]
    dof = len(cut_data_time) - (degree + 1)
    s2 = np.sum(residuals ** 2) / dof
    try:
        cov_matrix = np.linalg.inv(
            np.dot(np.transpose(np.vander(cut_data_time, degree + 1)), np.vander(cut_data_time, degree + 1))) * s2
        std_errors = np.sqrt(np.diag(cov_matrix))
    except LinAlgError:
        pass
    return [1.3 * (1 / np.power(10, 7)), 0.644, 10 ^ 5] if all(np.isnan(std_errors)) else std_errors.tolist()


def parabolic_fit(cut_data_time, cut_data_int, cut_data_int_err, I0):
    degree = 2
    weights = 1.0 / cut_data_int_err ** 2

    coefficients, residuals, rank, singular_values, rcond = np.polyfit(cut_data_time, cut_data_int, degree, w=weights,
                                                                       full=True)
    poly_function = np.poly1d(coefficients)
    parabolic = poly_function(cut_data_time)

    dof = len(cut_data_time) - (degree + 1)
    chi2_red = np.sum(((cut_data_int - parabolic) / cut_data_int_err) ** 2) / dof

    a2, a1, a0 = coefficients[0], coefficients[1], coefficients[2]
    residuals = cut_data_int - parabolic
    coefficients_errs = get_coefficient_errors(residuals, cut_data_time, degree)

    t_max = (-1) * a1 / (2 * a2)
    m_min = poly_function(t_max)
    umin, umin_err = umin_from_parabola(I0, m_min)
    tau, tau_err = tau_from_parabola(a2, coefficients_errs[0], a1, coefficients_errs[1], a0, coefficients_errs[2])

    # Propagate error
    t_max_err = np.sqrt(np.power((a1 / (2 * (a2 ** 2))) * coefficients_errs[0], 2) +
                        np.power((1 / (2 * a2)) * coefficients_errs[1], 2))

    return dict(tau=tau, tau_err=tau_err, a2=a2, a1=a1, a0=a0, a0_err=coefficients_errs[2], chi2=chi2_red, Tmax=t_max,
                Tmax_err=t_max_err, umin=umin, umin_err=umin_err)
