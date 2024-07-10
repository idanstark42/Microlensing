from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import chi2
from src.settings import BINS, PPFS


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


def plot_full_fit(data, fit):
  time = [point['t'] for point in data]
  I = [point['I'].value for point in data]
  I_err = [point['I'].error for point in data]
  plt.scatter(time, I, label="Data", s=1)
  plt.errorbar(time, I, yerr=I_err, fmt='o')
  x = np.linspace(min(time), max(time))
  y = fit(x)
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
    fig.colorbar(cp, label=r'$\chi^2$')

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

  ax.set_xlabel(key1)
  ax.set_ylabel(key2)
  ax.set_title(f"χ² ({key1} / {key2})")

  if independent:
    plt.show()


# multidimensional corner plots with different possible plots

def corner_plot(values, dimensions):
  min_ind = np.argmin(np.array([values[idx]['chi2'] for idx, _ in np.ndenumerate(values)]))
  min_key = list(np.ndenumerate(values))[min_ind][0]
  key_pairs = [('umin', 'Tmax'), ('umin', 'tau'), ('umin', 'fbl'),
         ('Tmax', 'tau'), ('Tmax', 'fbl'), ('tau', 'fbl')]
  fig, axes = plt.subplots(len(key_pairs), len(key_pairs), figsize=(15, 15))
  axes = np.array(axes)
  for i, variables in enumerate(key_pairs):
    ax = axes[i // len(key_pairs), i % len(key_pairs)]
    plot_chi_squared_map_contour(values, dimensions, variables=variables, const_indices=min_key, ax=ax)
  plt.tight_layout()
  plt.show()

