import numpy as np

# dimensions: dictionary of dimensions, each dimension is a tuple of (center, width, resolution), the key is the name of the variable

def generate_chi_squared_nd_map (dimensions, data, fit, dof):
  values = generate_values_nd(dimensions)
  it = np.nditer(values, flags=['multi_index', 'refs_ok'])
  for _ in it:
    values[it.multi_index]['chi2'] = chi_squared_reduced(data, fit, dof)
  return values

def chi_squared_reduced (data, fit, dof):
  return sum([(point['I'].value - fit(point['t']) / point['I'].error) ** 2 for i, point in enumerate(data)]) / dof

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
