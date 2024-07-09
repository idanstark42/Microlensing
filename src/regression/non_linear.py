import matplotlib.pyplot as plt
import numpy as np

from src.settings import PPFS
from src.utils import Value

# dimensions: dictionary of dimensions, each dimension is a tuple of (center, width, resolution), the key is the name
# of the variable

def generate_chi_squared_nd_map(dimensions, data, get_fit, dof):
  time = np.array(list(point['t'] for point in data))
  values = generate_values_nd(dimensions)
  it = np.nditer(values, flags=['multi_index', 'refs_ok'])
  for _ in it:
    fit = get_fit(time, values[it.multi_index]['umin'], values[it.multi_index]['Tmax'])
    values[it.multi_index]['chi2'] = chi_squared_reduced(data, fit, dof)

  min_ind = np.argmin(np.array([values[idx]['chi2'] for idx, _ in np.ndenumerate(values)]))
  min_key = list(np.ndenumerate(values))[min_ind][0]
  min_chi2 = values[min_key]['chi2']

  def get_error(key):
    min_value = values[min_key][key]
    array_index = [slice(None) for _ in range(len(dimensions))]
    array_index[list(dimensions.keys()).index(key)] = min_key[list(dimensions.keys()).index(key)]
    array_index = tuple(array_index)
    relevant_values = values[array_index]

    lower_section = [value for value in relevant_values if value[key] < min_value]
    upper_section = [value for value in relevant_values if value[key] > min_value]
    lower_index = np.argmin(np.array([abs(abs(value['chi2'] - min_chi2) - PPFS[dof][0]) for value in lower_section]))
    upper_index = np.argmin(np.array([abs(abs(value['chi2'] - min_chi2) - PPFS[dof][0]) for value in upper_section]))
    return (abs(min_value - lower_section[lower_index][key]), abs(min_value - upper_section[upper_index][key]))

  params = { key: Value(values[min_key][key], get_error(key)) for key in dimensions.keys() }
  params['chi2'] = min_chi2

  return values, params

def chi_squared_reduced(data, fit, dof):
  values = np.array(list(point['I'].value for point in data))
  errors = np.array(list(point['I'].error for point in data))
  return sum(((values - fit) / errors) ** 2) / (dof+len(data))


def generate_values_nd(dimensions):
  vectors = [generate_values_1d(*dimension) for dimension in dimensions.values()]
  grids = np.meshgrid(*vectors)
  combined = np.empty(grids[0].shape, dtype=object)
  it = np.nditer(combined, flags=['multi_index', 'refs_ok'])
  for _ in it:
    combined[it.multi_index] = { key: grids[i][it.multi_index] for i, key in enumerate(dimensions.keys()) }
  return combined


def generate_values_1d(center, width, resolution):
  return np.linspace(center - width / 2, center + width / 2, resolution)
