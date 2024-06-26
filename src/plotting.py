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