from matplotlib import pyplot as plt
import numpy as np

BINS = 50

def plot_event(event):
  time = [point['t'] for point in event.data]
  I = [point['I'].value for point in event.data]
  I_err = [point['I'].error for point in event.data]

  plt.scatter(time, I)
  plt.errorbar(time, I, yerr=I_err, fmt='o')
  plt.title(f"event {event.year}/{event.id}")
  plt.xlabel('t[sec]')
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
  plt.xlabel('t[sec]')
  plt.ylabel('I/I0')
  plt.show()

def plot_histogram_and_gaussians(samples, name):
  plt.hist(samples, bins=BINS, density=True, alpha=0.6)
  mean = np.mean(samples)
  std = np.std(samples)
  x = np.linspace(min(samples), max(samples), 100)
  y = 1 / (std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mean) / std) ** 2)
  plt.plot(x, y, label="Gaussian")
  plt.title(f"{name} histogram")
  plt.xlabel(name)
  plt.ylabel('# of samples')
  plt.show()

# chi squared map plotting

def plot_chi_squared_map_gridmap (values, dimensions, independent=True):
  plot_chi_squared_map(values, dimensions, lambda x, y, z: plt.matshow(z, extent=(min(x), max(x), min(y), max(y)), aspect='auto'), independent)

def plot_chi_squared_map_contour (values, dimensions, independent=True):
  plot_chi_squared_map(values, dimensions, plt.contour, independent)

def plot_chi_squared_map (values, dimensions, method, independent=True):
  key1, key2 = list(dimensions.keys())
  (center1, width1, resolution1), (center2, width2, resolution2) = dimensions[key1], dimensions[key2]
  x = np.linspace(center1 - width1 / 2, center1 + width1 / 2, resolution1)
  y = np.linspace(center2 - width2 / 2, center2 + width2 / 2, resolution2)
  z = np.array([[values[(x, y)]['chi2'] for x in x] for y in y])
  method(x, y, z)
  if independent:
    plt.xlabel(key1)
    plt.ylabel(key2)
    plt.title(f"χ² ({key1} / {key2})")
    plt.show()

# multidimensional corner plots with different possible plots

def corner_plot (values, dimensions, plot):
  keys = list(dimensions.keys())
  n = len(keys)
  retarded_keys = [keys[i+1] for i in range(n-1)]
  advanced_keys = [keys[i-1] for i in range(1, n)]
  fig, axs = plt.subplots(n, n, figsize=(15, 15))
  for i in range(n):
    for j in range(n):
      if i > j:
        axs[i, j].axis('off')
      else:
        key1 = retarded_keys[i], key2 = advanced_keys[i]
        plot(axs[i, j], values, key1, key2)