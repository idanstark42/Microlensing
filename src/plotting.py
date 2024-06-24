from matplotlib import pyplot as plt
import numpy as np

BINS = 30

def plot_data_and_parabola (data, predication):
  plt.scatter([point['t'] for point in data], [point['m'].value for point in data])
  plt.errorbar([point['t'] for point in data], [point['m'].value for point in data], yerr=[point['m'].error for point in data], fmt='o')
  x = np.linspace(min([point['t'] for point in data]), max([point['t'] for point in data]), 100)
  y = predication['a2'].value * x ** 2 + predication['a1'].value * x + predication['a0'].value
  plt.plot(x, y)
  plt.show(block=False)

def plot_histogram_and_gaussians (samples, values, name):
  plt.hist(samples, bins=BINS, density=True)
  for value in values:
    x = np.linspace(min(samples), max(samples), 100)
    y = 1 / (value.error * (2 * np.pi) ** 0.5) * np.exp(- (x - value.value) ** 2 / (2 * value.error ** 2))
    plt.plot(x, y)
  plt.title(name)
  plt.show(block=False)