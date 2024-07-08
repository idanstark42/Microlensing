from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import chi2

from src.settings import BINS


def plot_event(event):
    time = [point['t'] for point in event.data]
    I = [point['I'].value for point in event.data]
    I_err = [point['I'].error for point in event.data]

    plt.scatter(time, I)
    plt.errorbar(time, I, yerr=I_err, fmt='o')
    plt.title(f"event {event.year}/{event.id}")
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


def plot_data_and_parabola(data, predication):
    time = [point['t'] for point in data]
    I = [point['I'].value for point in data]
    I_err = [point['I'].error for point in data]
    plt.scatter(time, I, label="Data")
    plt.errorbar(time, I, yerr=I_err, fmt='o')
    x = np.linspace(min(time), max(time), 100)
    y = predication['a2'].value * (x ** 2) + predication['a1'].value * x + predication['a0'].value
    plt.plot(x, y, label="Fitted Parabola")
    plt.xlabel('t[years]')
    plt.ylabel('I/I0')
    plt.show()


def plot_histogram_and_gaussian(samples, name, gaussian):
    plt.hist(samples, bins=BINS, density=True, alpha=0.6, edgecolor='black')
    x = np.linspace(min(samples), max(samples), 1000)
    y = gaussian(x)
    # plt.axvline(mean, color='blue', linestyle='--', label=f'Mean: {mean:.2f}')
    plt.plot(x, y, label="Gaussian")
    plt.title(f"{name} histogram")
    plt.xlabel(name)
    plt.ylabel('# of samples')
    plt.show()


# chi squared map plotting

def plot_chi_squared_map_gridmap(values, dimensions, independent=True):
    plot_chi_squared_map(values, dimensions,
                         lambda x, y, z: plt.matshow(z, extent=(min(x), max(x), min(y), max(y)), aspect='auto'),
                         independent)


def plot_chi_squared_map_contour(values, dimensions, independent=True):
    plot_chi_squared_map(values, dimensions, plt.contour, independent)


def plot_chi_squared_map(values, dimensions, method, independent=True):
    print(values.shape)
    #umin in Y, tmax in X
    tmax, umin = list(dimensions.keys())
    (tmax_center, tmax_width, tmax_resolution), (umin_center, umin_width, umin_resolution) = dimensions[tmax], dimensions[umin]
    x = np.linspace(tmax_center - tmax_width / 2, tmax_center + tmax_width / 2, tmax_resolution)
    y = np.linspace(umin_center - umin_width / 2, umin_center + umin_width / 2, umin_resolution)
    it = np.nditer(values, flags=['multi_index', 'refs_ok'])
    chi2_min = np.min(np.array([values[it.multi_index]['chi2'] for _ in it]))
    chi2_values = np.array([[d['chi2'] for d in row] for row in values])
    print(chi2_values.shape)
    print(values.shape)
    print(f"Min Chi2: {chi2_min}")
    chi2_flattened_values = np.array([d['chi2'] for d in values.flatten()])
    min_index = np.unravel_index(np.argmin(chi2_flattened_values), values.shape)
    print("Minimum chi2 position:", min_index)
    delta_chi2_1sigma = chi2.ppf(0.6827, df=2)
    delta_chi2_2sigma = chi2.ppf(0.9545, df=2)
    delta_chi2_3sigma = chi2.ppf(0.9973, df=2)
    levels = [chi2_min + delta_chi2_1sigma, chi2_min + delta_chi2_2sigma, chi2_min + delta_chi2_3sigma]
    print("Contour levels for 1-sigma, 2-sigma, 3-sigma:", levels)
    plt.figure(figsize=(8, 6))
    cp = plt.contour(y, x, chi2_values.T, levels=levels, colors=['blue', 'green', 'red'],
                     linestyles=['solid', 'dashed', 'dashdot'])
    plt.clabel(cp, inline=True, fontsize=10, fmt={level: f'{sigma}σ' for level, sigma in zip(levels, ['1', '2', '3'])})
    plt.scatter(y[min_index[0]], x[min_index[1]], color='black', marker='x', label='Min $\chi^2$')
    plt.xlabel('Variable 2 (y)')
    plt.ylabel('Variable 1 (x)')
    plt.title(r'Contour plot of $\chi^2$ with confidence levels')
    plt.legend()
    plt.colorbar(cp, label=r'$\chi^2$')
    plt.show()

    # z = np.array([[values[(x_val, y_val)]['chi2'] for y_val in range(0, 200)] for x_val in range(0, 100)])
    # print(z.shape)
    # if independent:
    #     if method == plt.contour:
    #         chi2_min = np.min(z)
    #         min_index = np.unravel_index(np.argmin(z), z.shape)
    #         # Plotting
    #         plt.scatter(x[min_index[1]], y[min_index[0]], color='red', marker='x', label='Min $\chi^2$')
    #
    #         # Confidence levels based on chi-squared distribution
    #
    #         plt.contour(y, x, z.T, levels=[chi2_min + level for level in [1, 3, 4]], colors='black', linestyles='solid')
    #         plt.ylim(0, 0.5)
    #
    #         # chi2_min = np.min(z)
    #         # min_index = np.unravel_index(np.argmin(z), z.shape)
    #         # print(chi2_min)
    #         # print(z[min_index[0],min_index[1]])
    #         # plt.contourf(y, x, z.T, levels=100, cmap='viridis')
    #         # plt.colorbar(label=r'$\chi^2$')
    #         # plt.scatter(y[min_index[1]], x[min_index[0]], color='red', marker='x', label='Min $\chi^2$')
    #         # # Plot confidence ellipses
    #         # delta_chi2 = z - chi2_min
    #         # confidence_levels = [chi2.ppf(level, 2) for level in [0.6827, 0.9545, 0.9973]]
    #         # for level in confidence_levels:
    #         #     plt.contour(y, x, delta_chi2, levels=[level], colors='black', linestyles='solid')
    #     else:
    #         method(x, y, z)
    #         plt.colorbar()
    #     plt.xlabel(key1)
    #     plt.ylabel(key2)
    #     plt.title(f"χ² ({key1} / {key2})")
    #     plt.legend()
    #     plt.show()


# multidimensional corner plots with different possible plots

def corner_plot(values, dimensions, plot):
    keys = list(dimensions.keys())
    n = len(keys)
    retarded_keys = [keys[i + 1] for i in range(n - 1)]
    advanced_keys = [keys[i - 1] for i in range(1, n)]
    fig, axs = plt.subplots(n, n, figsize=(15, 15))
    for i in range(n):
        for j in range(n):
            if i > j:
                axs[i, j].axis('off')
            else:
                key1 = retarded_keys[i], key2 = advanced_keys[i]
                plot(axs[i, j], values, key1, key2)
