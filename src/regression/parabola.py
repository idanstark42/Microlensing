import numpy as np
from numpy.linalg import LinAlgError
from src.utils import Value, u

def fit_parabola(data, degree=2):
  time = np.array([point['t'] for point in data])
  I = np.array([point['I'].value for point in data])
  I_err = np.array([point['I'].error for point in data])

  weights = 1.0 / I_err ** 2
  coefficients, residuals, rank, singular_values, rcond = np.polyfit(time, I, degree, w=weights, full=True)
  parabolic = np.poly1d(coefficients)(time)

  dof = len(time) - (degree + 1)
  residuals = I - parabolic
  chi2_red = np.sum((residuals / I_err) ** 2) / dof

  coefficients_errs = calc_coefficient_errors(residuals, time, degree)
  a2, a1, a0 = [Value(coefficients[i], coefficients_errs[i]) for i in range(degree + 1)]

  tau = calc_tau(a0, a1, a2)
  t_max = calc_tmax(a0, a1, a2)
  umin = calc_umin(a0, a1, a2, t_max)

  return { 'tau': tau, 'Tmax': t_max, 'umin': umin, 'a2': a2, 'a1': a1, 'a0': a0, 'chi2': chi2_red }

# helper functions

def calc_tmax(a0, a1, a2):
  t_max = (-1) * a1.value / (2 * a2.value)
  t_max_error = np.sqrt(np.power((a1.value / (2 * (a2.value ** 2))) * a2.error, 2) + np.power((1 / (2 * a2.value)) * a1.error, 2))
  return Value(t_max, t_max_error)

def calc_tau(a0, a1, a2):
  tau = np.sqrt(np.abs((a1.value ** 2 - 4 * a2.value * a0.value) / 2)) / abs(a2.value)
  tau_error = (1 / tau) * (((a1.value ** 2 + 4 * a2.value * a0.value) / 4 * a2.value ** 3) * a2.error + (a1.value / a2.value ** 2) * a1.error - (2 / a2.value) * a0.error)
  return Value(tau, tau_error)

def calc_umin(a0, a1, a2, t_max):
  I_max = a2.value * t_max.value ** 2 + a1.value * t_max.value + a0.value
  I_max_error = (((t_max.value ** 2) * a2.error) ** 2 + (t_max.value * a1.error) ** 2 + a0.error ** 2 + ((2 * a2.value * t_max.value + a1.value) * t_max.error) ** 2) ** 0.5
  return u(Value(I_max, I_max_error))

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
