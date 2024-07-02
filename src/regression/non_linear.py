import numpy as np
from iminuit import Minuit
from iminuit.cost import LeastSquares


def generate_chi_squared_nd_map(center, width, resolution, data, fit, dof):
    values = generate_values_nd(center, width, resolution)
    it = np.nditer(values, flags=['multi_index', 'refs_ok'])
    for _ in it:
        values[it.multi_index]['chi2'] = chi_squared_reduced(data, fit, dof)


def chi_squared_reduced(data, fit, dof):
    residuals = [point['I'].value - fit(point['t']) for point in data]
    return sum([(residuals[i] / point['I'].error) ** 2 for i, point in enumerate(data)]) / dof


def generate_values_nd(dimensions):
    vectors = [generate_values_1d(*dimension) for dimension in dimensions.values()]
    grids = np.meshgrid(*vectors)
    combined = np.empty(grids[0].shape, dtype=object)
    it = np.nditer(combined, flags=['multi_index', 'refs_ok'])
    for _ in it:
        combined[it.multi_index] = {key: grids[i][it.multi_index] for i, key in enumerate(dimensions.keys())}
    return combined


def generate_values_1d(center, width, resolution):
    return np.linspace(center - width / 2, center + width / 2, resolution)


def gaussian(x, amplitude, mean, sigma):
    return amplitude / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def fit_gaussian(bin_midpoints, bin_counts, error):
    least_squares = LeastSquares(bin_midpoints, bin_counts, 1, gaussian)
    m = Minuit(least_squares, amplitude=0.7, mean=np.mean(bin_midpoints), sigma=np.std(bin_midpoints))
    m.migrad()
    print(f"χ²:\t{m.fval/(len(bin_midpoints)-3)}")
    return m.values["amplitude"], m.values["mean"], m.values["sigma"]
