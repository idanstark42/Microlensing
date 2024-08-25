import numpy as np
from numpy.linalg import LinAlgError
from src.utils import Value, u
from iminuit import Minuit
from iminuit.cost import LeastSquares
from matplotlib import pyplot as plt

from src.settings import BINS


# polynomial

def fit_polynomial(data, degree=2):
    t_shift = data[0]['t']
    time = np.array([point['t'] - t_shift for point in data])
    I = np.array([point['I'].value for point in data])
    I_err = np.array([point['I'].error for point in data])

    weights = 1.0 / I_err ** 2
    coefficients, residuals, rank, singular_values, rcond = np.polyfit(time, I, degree, w=weights, full=True)
    fit = np.poly1d(coefficients)(time)

    dof = len(time) - (degree + 1)
    residuals = I - fit
    
    chi2_red = np.sum((residuals / I_err) ** 2) / dof

    coefficients_errs = calc_coefficient_errors(residuals, time, degree)
    a2, a1, a0 = [Value(coefficients[i], coefficients_errs[i]) for i in range(degree + 1)]

    if a2.value > 0 or a0.value < 0:
        return None

    t_max = calc_tmax(a2, a1, a0)
    umin = calc_umin(a2, a1, a0, t_max)
    tau = calc_tau(a2, a1, a0, umin)
    t_max.value += t_shift
    
    shifted_a2_value = a2.value
    shifted_a1_value = a1.value - 2 * a2.value * t_shift
    shifted_a0_value = a0.value + a2.value * t_shift ** 2 - a1.value * t_shift

    a2, a1, a0 = Value(shifted_a2_value, a2.error), Value(shifted_a1_value, a1.error), Value(shifted_a0_value, a0.error)


    return { 'tau': tau, 'Tmax': t_max, 'umin': umin, 'a2': a2, 'a1': a1, 'a0': a0, 'chi2': chi2_red, 'time': time, 'residuals': residuals }

# helper functions

def calc_tmax(a2, a1, a0):
    t_max = (-1) * a1.value / (2 * a2.value)
    t_max_error = ((t_max * a2.error / a2.value) ** 2 + (t_max * a1.error / a1.value) ** 2) ** 0.5
    return Value(t_max, t_max_error)


def calc_umin(a2, a1, a0, t_max):
    I_max = a2.value * t_max.value ** 2 + a1.value * t_max.value + a0.value
    I_max_error = (((t_max.value ** 2) * a2.error) ** 2 + (t_max.value * a1.error) ** 2 + a0.error ** 2 + (
            (2 * a2.value * t_max.value + a1.value) * t_max.error) ** 2) ** 0.5
    return u(Value(I_max, I_max_error))


def calc_tau(a2, a1, a0, umin):
    I_tau = (umin.value ** 2 + 2.25) / (((umin.value ** 2 + 0.25) * (umin.value ** 2 + 4.25)) ** 0.5)
    I_tau_error = 8 * umin.value / (((umin.value ** 2 + 0.25) * (umin.value ** 2 + 4.25)) ** 1.5) * umin.error
    descriminant = a1.value ** 2 - 4 * a2.value * (a0.value - I_tau)
    tau = descriminant ** 0.5 / np.abs(a2.value)
    
    tau_error = tau * (((a1.value ** 2) * (a1.error ** 2) + 4 * (a2.value ** 2) *(I_tau_error ** 2 + a0.error ** 2) + descriminant * (a2.error ** 2) / (a2.value ** 2)) / descriminant) ** 0.5
    return Value(tau, tau_error)

def calc_coefficient_errors(residuals, time, degree):
    std_errors = [np.nan, np.nan, np.nan]
    dof = len(time) - (degree + 1)
    s2 = np.sum(residuals ** 2) / dof
    try:
        cov_matrix = np.linalg.inv(
            np.dot(np.transpose(np.vander(time, degree + 1)), np.vander(time, degree + 1))) * s2
        std_errors = np.sqrt(np.diag(cov_matrix))
    except LinAlgError:
        pass
    return [1.3 * (1 / np.power(10, 7)), 0.644, 10 ^ 5] if all(np.isnan(std_errors)) else std_errors.tolist()


# gaussian

def gaussian(x, amplitude, mean, sigma):
    return amplitude / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)

def fit_histogram_gaussian(samples):
    bin_counts, bin_edges = np.histogram(samples, bins=BINS, density=True)
    bin_midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2
    least_squares = LeastSquares(bin_midpoints, bin_counts, 1, gaussian)
    m = Minuit(least_squares, amplitude=1, mean=np.mean(bin_midpoints), sigma=np.std(bin_midpoints))
    m.migrad()
    return m.values["amplitude"], m.values["mean"], m.values["sigma"], m.fval / (len(bin_midpoints) - 3), bin_midpoints, bin_counts - gaussian(bin_midpoints, m.values["amplitude"], m.values["mean"], m.values["sigma"])
