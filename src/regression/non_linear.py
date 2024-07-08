import matplotlib.pyplot as plt
import numpy as np


# dimensions: dictionary of dimensions, each dimension is a tuple of (center, width, resolution), the key is the name
# of the variable

def generate_chi_squared_nd_map(dimensions, data, get_fit, dof):
    I_shift = np.min(np.array([datum['I'].value for datum in data]))
    for datum in data:
        datum['I'].value -= I_shift
    values = generate_values_nd(dimensions)
    it = np.nditer(values, flags=['multi_index', 'refs_ok'])
    for _ in it:
        fit = get_fit(values[it.multi_index]['umin'], values[it.multi_index]['Tmax'])
        shifted_fit = fit - np.min(fit)
        values[it.multi_index]['chi2'] = chi_squared_reduced(data, shifted_fit, dof)

    chi2_values = np.array([values[idx]['chi2'] for idx, _ in np.ndenumerate(values)])
    min_ind = np.argmin(chi2_values)
    min_key = list(np.ndenumerate(values))[min_ind][0]
    min_value = values[min_key]
    print(f"min chi2 = {min_value['chi2']}")

    print("Showing the minimal X2 fit")
    fit2 = get_fit(min_value['umin'], min_value['Tmax'])
    shifted_fit2 = fit2-min(fit2)
    plt.plot(shifted_fit2)
    plt.plot(np.array([datum['I'].value for datum in data]))
    plt.show()
    return values


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
        combined[it.multi_index] = {key: grids[i][it.multi_index] for i, key in enumerate(dimensions.keys())}
    return combined


def generate_values_1d(center, width, resolution):
    return np.linspace(center - width / 2, center + width / 2, resolution)
