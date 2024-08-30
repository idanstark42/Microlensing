from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import chi2
from src.settings import BINS, PPFS, TIME_WINDOW


def plot_event(event):
    time = [point['t'] for point in event.data]
    I = [point['I'].value for point in event.data]
    I_err = [point['I'].error for point in event.data]

    plt.scatter(time, I)
    plt.errorbar(time, I, yerr=I_err, fmt='o')
    plt.title(f"event {event.year}/{event.id}")
    plt.axvline(event.metadata['Tmax'].value, color='red', linestyle='--', label='Tmax')
    plt.axvline(event.metadata['Tmax'].value - 100, color='green', linestyle='--', label='Tmax - 100')
    plt.axvline(event.metadata['Tmax'].value + 100, color='green', linestyle='--', label='Tmax + 100')
    plt.axhline(event.metadata['I*'].value, color='blue', linestyle='--', label='Time Window')
    plt.xlabel('t[day]')
    plt.ylabel('I/I0')
    plt.show()


def plot_data(data):
    time = [point['t'] for point in data]
    I = [point['I'].value for point in data]
    I_err = [point['I'].error for point in data]
    plt.scatter(time, I)
    plt.errorbar(time, I, yerr=I_err, fmt='o')
    plt.xlabel('t[day]')
    plt.ylabel('I/I0')
    plt.show()

def plot_residuals(x, residuals, title="Residuals", xlabel="t [days]"):
    plt.scatter(x, residuals)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Residuals')
    # add line at 0
    plt.axhline(0, color='black', linestyle='--')
    plt.show()

def plot_data_and_parabola(data, prediction, title="Data and Parabola"):
    time = [point['t'] for point in data]
    I = [point['I'].value for point in data]
    I_err = [point['I'].error for point in data]
    plt.scatter(time, I, label="Data")
    plt.errorbar(time, I, yerr=I_err, fmt='o')
    max_time = - prediction['a1'].value / (2 * prediction['a2'].value)
    tau = prediction['tau'].value
    x = np.linspace(max_time - tau, max_time + tau, 100)
    y = prediction['a2'].value * (x ** 2) + prediction['a1'].value * x + prediction['a0'].value
    plt.plot(x, y, label="Fitted Parabola")
    plt.axvline(max_time + tau / 2, color='blue', linestyle='--')
    plt.axvline(max_time - tau / 2, color='blue', linestyle='--')
    plt.title(title)
    plt.xlabel('t [days]')
    plt.ylabel('I/I0')
    plt.ylim(min(I) - 0.1, max(I) + 0.1)
    plt.show()


def plot_histogram_and_gaussian(samples, name, gaussian, units):
    plt.hist(samples, bins=BINS, alpha=0.6, edgecolor='black', density=True)
    x = np.linspace(min(samples), max(samples), 1000)
    y = gaussian(x)
    mean = np.mean(samples)
    plt.axvline(mean, color='blue', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.plot(x, y, label="Gaussian")
    plt.title(f"{name} Histogram")
    plt.xlabel(name + (f' [{units}]' if units else ''))
    plt.ylabel('Probability Density')
    plt.show()


def plot_full_fit(data, fit):
    time = [point['t'] for point in data]
    I = [point['I'].value for point in data]
    I_err = [point['I'].error for point in data]
    plt.scatter(time, I, label="Data", s=1)
    plt.errorbar(time, I, yerr=I_err, fmt='o')
    x = np.linspace(min(time), max(time))
    y = fit(x)
    plt.title("I(t)/I* Fit")
    plt.xlabel('t [days]')
    plt.ylabel('I/I0')
    plt.plot(x, y, label="Fit")
    plt.show()


# chi squared map plotting

def plot_chi_squared_map_gridmap(values, dimensions, variables=None, const_indices=(), ax=None):
    def plot(x, y, z, ax, fig):
        ax.matshow(z, extent=(min(x), max(x), min(y), max(y)), aspect='auto', cmap='viridis')
        fig.colorbar(ax.matshow(z, extent=(min(x), max(x), min(y), max(y)), aspect='auto', cmap='viridis'),
                     label=r'$\chi^2$')

    plot_chi_squared_map(values, dimensions, plot, variables, const_indices, ax)


def plot_chi_squared_map_contour(values, dimensions, variables=None, const_indices=(), ax=None, dof=2):
    def plot(x, y, z, ax, fig):
        it = np.nditer(z, flags=['multi_index', 'refs_ok'])
        z_min = np.min(np.array([z[it.multi_index] for _ in it]))
        z_flattened = np.array(z.flatten())
        min_index = np.unravel_index(np.argmin(z_flattened), z.shape)
        levels = [(index + 1, z_min + chi2.ppf(ppf, df=dof)) for index, ppf in enumerate(PPFS)]
        cp = ax.contour(x, y, z, levels=[level[1] for level in levels], colors=['blue', 'green', 'red'],
                        linestyles=['solid', 'dashed', 'dashdot'])
        ax.clabel(cp, inline=True, fontsize=10, fmt={level: f'{sigma}σ' for sigma, level in levels})
        ax.scatter(x[min_index[1]], y[min_index[0]], color='black', marker='x', label='Min $\chi^2$')
        ax.legend()
        # fig.colorbar(cp, label=r'$\chi^2$')

    plot_chi_squared_map(values, dimensions, plot, variables, const_indices, ax)


def plot_chi_squared_map(values, dimensions, method, variables, const_indices, ax):
    if variables is None:
        variables = list(dimensions.keys())
    if (len(variables) != 2):
        raise ValueError("Only 2D plots are supported")

    key1, key2 = variables

    (center1, width1, resolution1), (center2, width2, resolution2) = dimensions[key1], dimensions[key2]
    x = np.linspace(center1 - width1 / 2, center1 + width1 / 2, resolution1)
    y = np.linspace(center2 - width2 / 2, center2 + width2 / 2, resolution2)
  
    array_index = []
    keys_to_keep = [list(dimensions.keys()).index(key1), list(dimensions.keys()).index(key2)]
    for dim in range(len(dimensions)):
        array_index.append(slice(None) if dim in keys_to_keep else const_indices[dim])
    array_index = tuple(array_index)
    relevant_values = values[array_index]
    z = np.array([[d['chi2'] for d in row] for row in relevant_values])

    independent = ax is None
    if independent:
        ax = plt.gca()
        fig = plt.gcf()
    else:
        fig = ax.get_figure()

    method(x, y, z, ax, fig)

    ax.set_xlabel(key1, fontsize=8)
    ax.set_ylabel(key2, fontsize=8)
    ax.set_title(f"χ² ({key1} / {key2})", fontsize=10)

    if independent:
        plt.show()


# multidimensional corner plots with different possible plots
def corner_plot(values, dimensions):
    min_ind = np.argmin(np.array([values[idx]['chi2'] for idx, _ in np.ndenumerate(values)]))
    min_key = list(np.ndenumerate(values))[min_ind][0]
    keys = ['umin', 'Tmax', 'tau', 'fbl']
    num_keys = len(keys)
    fig, axes = plt.subplots(num_keys, num_keys, figsize=(15, 15))
    axes = np.array(axes)
    for i in range(1, num_keys):
        for j in range(i):
            var_x = keys[j]
            var_y = keys[i]
            ax = axes[i, j]
            if (var_x=='tau' and var_y=='fbl') or (var_y=='tau' and var_x=='fbl'):
                plot_chi_squared_map_contour(values, dimensions, variables=(var_x, var_y), const_indices=min_key, ax=ax)
            else:
                plot_chi_squared_map_contour(values, dimensions, variables=(var_y, var_x), const_indices=min_key, ax=ax)
    for i in range(num_keys):
        for j in range(num_keys):
            if i <= j:
                axes[i, j].set_visible(False)
    plt.suptitle('BLG 109 - Corner Plot', fontsize=16)
    plt.subplots_adjust(hspace=0.8, wspace=1)
    plt.tight_layout()
    plt.show()