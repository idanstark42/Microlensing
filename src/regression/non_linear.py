import numpy as np

def generate_chi_squared_nd_map (center, width, resolution, data, fit, dof):
  values = generate_values_nd(center, width, resolution)
  it = np.nditer(values, flags=['multi_index', 'refs_ok'])
  for _ in it:
    values[it.multi_index]['chi2'] = chi_squared_reduced(data, fit, dof)

def chi_squared_reduced (data, fit, dof):
  residuals = [point['I'].value - fit(point['t']) for point in data]
  return sum([(residuals[i] / point['I'].error) ** 2 for i, point in enumerate(data)]) / dof

def generate_values_nd (dimensions):
  vectors = [generate_values_1d(*dimension) for dimension in dimensions.values()]
  grids = np.meshgrid(*vectors)
  combined = np.empty(grids[0].shape, dtype=object)
  it = np.nditer(combined, flags=['multi_index', 'refs_ok'])
  for _ in it:
    combined[it.multi_index] = { key: grids[i][it.multi_index] for i, key in enumerate(dimensions.keys()) }
  return combined

def generate_values_1d (center, width, resolution):
  return np.linspace(center - width / 2, center + width / 2, resolution)