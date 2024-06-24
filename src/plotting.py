from matplotlib import pyplot as plt
import numpy as np

from utils import Value

BINS = 30

def plot_data_and_parabola (data, predication):
  plt.scatter([point['t'] for point in data], [point['m'].value for point in data])
  plt.errorbar([point['t'] for point in data], [point['m'].value for point in data], yerr=[point['m'].error for point in data], fmt='o')
  x = np.linspace(min([point['t'] for point in data]), max([point['t'] for point in data]), 100)
  y = predication['a2'].value * x ** 2 + predication['a1'].value * x + predication['a0'].value
  plt.plot(x, y)
  plt.show()

def plot_histogram_and_gaussians (samples, values, name):
  plt.hist(samples, bins=BINS, density=True)
  for value in values:
    x = np.linspace(min(samples), max(samples), 100)
    y = 1 / (value.error * (2 * np.pi) ** 0.5) * np.exp(- (x - value.value) ** 2 / (2 * value.error ** 2))
    plt.plot(x, y)
  plt.title(name)
  plt.show()


if __name__ == '__main__':
  plot_histogram_and_gaussians([1, 2, 3, 4, 5, 3, 4, 2, 3, 2, 4, 3, 3], [Value(3, 1)], 'test')