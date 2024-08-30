import numpy as np
from scipy.stats import chi2
from progress.bar import IncrementalBar

from src.settings import PPFS
from src.utils import Value

# dimensions: dictionary of dimensions, each dimension is a tuple of (center, width, resolution), the key is the name
# of the variable

def generate_chi_squared_nd_map(dimensions, data, get_fit, dof, use_bar=True):
  time = np.array(list(point['t'] for point in data))
  values = generate_values_nd(dimensions)
  it = np.nditer(values, flags=['multi_index', 'refs_ok'])
  if use_bar:
    bar = IncrementalBar('Calculating chi squared', max=values.size, suffix='%(percent)d%%')
  for _ in it:
    fit = get_fit(time, values[it.multi_index])
    values[it.multi_index]['chi2'] = chi_squared_reduced(data, fit, dof)
    if use_bar:
      bar.next()

  min_ind = np.argmin(np.array([values[idx]['chi2'] for idx, _ in np.ndenumerate(values)]))
  min_key = list(np.ndenumerate(values))[min_ind][0]
  min_chi2 = values[min_key]['chi2']

  def get_error(key):
    try:
      min_value = values[min_key][key]
      array_index = []
      key_to_keep = list(dimensions.keys()).index(key)
      for dim in range(len(dimensions)):
        array_index.append(slice(None) if dim == key_to_keep else min_key[dim])
      array_index = tuple(array_index)
      relevant_values = values[array_index]

      lower_section = [value for value in relevant_values if value[key] < min_value]
      upper_section = [value for value in relevant_values if value[key] > min_value]
      lower_index = np.argmin(
        np.array([abs(abs(value['chi2'] - min_chi2) - chi2.ppf(PPFS[0], df=1)) for value in lower_section]))
      upper_index = np.argmin(
        np.array([abs(abs(value['chi2'] - min_chi2) - chi2.ppf(PPFS[0], df=1)) for value in upper_section]))
      return abs(min_value - lower_section[lower_index][key]), abs(min_value - upper_section[upper_index][key])
    except IndexError:  # TODO Fix that
      return 0
    except ValueError:
      return 0

  params = {key: Value(values[min_key][key], get_error(key)) for key in dimensions.keys()}
  params['chi2'] = min_chi2
  if use_bar:
    bar.finish()
  return values, params


def chi_squared_reduced(data, fit, dof):
  values = np.array(list(point['I'].value for point in data))
  errors = np.array(list(point['I'].error for point in data))
  return sum(((values - fit) / errors) ** 2) / (dof + len(data))


def generate_values_nd(dimensions):
  vectors = [generate_values_1d(*dimension) for dimension in dimensions.values()]
  grids = np.meshgrid(*vectors, indexing='ij')
  combined = np.empty(grids[0].shape, dtype=object)
  it = np.nditer(combined, flags=['multi_index', 'refs_ok'])
  for _ in it:
    combined[it.multi_index] = {key: grids[i][it.multi_index] for i, key in enumerate(dimensions.keys())}
  return combined


def generate_values_1d(center, width, resolution):
  return np.linspace(center - width / 2, center + width / 2, resolution)
