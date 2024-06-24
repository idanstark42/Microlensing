import sys
from tabulate import tabulate

from src.regression import parabola
from src.bootstraping import bootstrap
from src.plotting import plot_data_and_parabola, plot_histogram_and_gaussians, show

from src.ogle import load_event

YEAR = '2019'
ID = 'blg-0001'
BOOTSTRAP_SAMPLES = 10000

def part_1():
  event = load_event(YEAR, ID)

  # TODO cut the data to the region of the peak
  cut_data = event.data
  predication = parabola(cut_data)

  tau_n_sigma = predication['tau'].n_sigma(event.metadata['tau'])
  umin_n_sigma = predication['umin'].n_sigma(event.metadata['umin'])
  tmax_n_sigma = predication['Tmax'].n_sigma(event.metadata['Tmax'])

  print('Finished first fit')
  print(tabulate([
    ['tau', predication['tau'], event.metadata['tau'], tau_n_sigma],
    ['umin', predication['umin'], event.metadata['umin'], umin_n_sigma],
    ['Tmax', predication['Tmax'], event.metadata['Tmax'], tmax_n_sigma]
  ], headers=['Parameter', 'Fit', 'OGLE', 'N Sigma']))
  
  
  bootstrap_predications = bootstrap(cut_data, lambda data: parabola(data), BOOTSTRAP_SAMPLES)

  plot_data_and_parabola(cut_data, predication)
  plot_histogram_and_gaussians(bootstrap_predications['tau'], [predication['tau'], event.metadata['tau']], 'tau')
  plot_histogram_and_gaussians(bootstrap_predications['umin'], [predication['umin'], event.metadata['umin']], 'umin')
  plot_histogram_and_gaussians(bootstrap_predications['Tmax'], [predication['Tmax'], event.metadata['Tmax']], 'Tmax')

def part_2():
  print('not implemented yet')


if __name__ == '__main__':
  command = sys.argv[1]

  if command == 'part1':
    part_1()

  elif command == 'part2':
    part_2()

  else:
    print('Invalid command')
    sys.exit(1)
