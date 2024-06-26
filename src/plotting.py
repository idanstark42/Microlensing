from matplotlib import pyplot as plt
import numpy as np
from .utils import Value

BINS = 50


def plot_data_and_parabola(cut_data_time, cut_data_int, cut_data_int_err, predication):
    plt.scatter(cut_data_time, cut_data_int, label="Data")
    plt.errorbar(cut_data_time, cut_data_int, yerr=cut_data_int_err, fmt='o')
    x = np.linspace(min(cut_data_time), max(cut_data_time), 100)
    y = predication['a2'] * (x ** 2) + predication['a1'] * x + predication['a0']
    plt.plot(x, y, label="Fitted Parabola")
    plt.xlabel('t[sec]')
    plt.ylabel('I magnitude')
    plt.show()


def plot_histogram_and_gaussians(samples, value, value_err, name):
    counts, bins, _ = plt.hist(samples, bins=BINS, density=True)

    mean = np.average(bins, weights=np.append(counts, 0))
    variance = np.average((bins - mean) ** 2, weights=np.append(counts, 0))
    std = np.sqrt(variance)

    bin_midpoint = bins[np.argmax(counts)]

    x1 = np.linspace(min(samples), max(samples), 100)
    y1 = max(counts) * np.exp(- (x1 - value) ** 2 / (2 * std ** 2))

    x2 = bins
    y2 = max(counts) * np.exp(- (bins - value) ** 2 / (2 * std ** 2))

    plt.figure(1)
    plt.plot(x1, y1)
    plt.title(name)

    plt.figure(2)
    plt.scatter(x2, y2-np.append(counts, 0), s=10, label='Residuals')
    plt.axhline(y=0, color='r', linestyle='--', label='y=0')
    plt.title(f"{name}: Residuals")
    plt.legend()
    plt.show()

    print(f"{name} (Value(hist)-Value(pred))/Value(hist) = {np.abs(value - bin_midpoint) / bin_midpoint}")


if __name__ == '__main__':
    plot_histogram_and_gaussians([1, 2, 3, 4, 5, 3, 4, 2, 3, 2, 4, 3, 3], [Value(3, 1)], 'test')
