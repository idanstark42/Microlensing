import sys
from tabulate import tabulate
import numpy as np
from src.regression import fit_parabola
from src.bootstraping import bootstrap
from src.plotting import plot_event, plot_data_and_parabola, plot_histogram_and_gaussians
from src.ogle import Event
from src.utils import I

YEAR = '2019'
ID = 'blg-0171'
BOOTSTRAP_SAMPLES = 10000
MIN_DATA_POINTS = 20
TIME_WINDOW = 7

def part_1():
  print('--- calulating parabolic fit ---')
  event = Event(YEAR, ID)
  data = event.points_around_peak(TIME_WINDOW)

  if len(data) < MIN_DATA_POINTS:
    print(f'Data has only {len(data)} points, which is less than the minimum of {MIN_DATA_POINTS}. Exiting.')
    sys.exit(1)

  print('1. Fitting parabola with all points')
  prediction = fit_parabola(data)
  print()
  print(f"χ²:\t{prediction['chi2']}")
  tau_n_sigma = prediction['tau'].n_sigma(event.metadata['tau'])
  umin_n_sigma = prediction['umin'].n_sigma(event.metadata['umin'])
  tmax_n_sigma = prediction['Tmax'].n_sigma(event.metadata['Tmax'])
  plot_data_and_parabola(data, prediction)

  print(tabulate([
    ['tau', prediction['tau'], event.metadata['tau'], tau_n_sigma],
    ['umin', prediction['umin'], event.metadata['umin'], umin_n_sigma],
    ['Tmax', prediction['Tmax'], event.metadata['Tmax'], tmax_n_sigma]
  ], headers=['Parameter', 'Fit', 'OGLE', 'N Sigma']))

  print('2. Bootstrapping')
  bootstrap_predictions = bootstrap(data, fit_parabola, BOOTSTRAP_SAMPLES)

  plot_histogram_and_gaussians([prediction['tau'].value for prediction in bootstrap_predictions], 'tau')
  plot_histogram_and_gaussians([prediction['umin'].value for prediction in bootstrap_predictions], 'umin')
  plot_histogram_and_gaussians([prediction['Tmax'].value for prediction in bootstrap_predictions], 'Tmax')
  print('done')

def part_2():
  print('not implemented yet')

def part_3():
  print('not implemented yet')

if __name__ == '__main__':
  command = sys.argv[1]

  if command == 'part1':
    part_1()

  elif command == 'part2':
    part_2()

  elif command == 'part3':
    part_3()

  elif command == 'event':
    print(Event(YEAR, ID))

  elif command == 'show_event':
    plot_event(Event(YEAR, ID))

  elif command == 'good_events':
    number_of_events = int(sys.argv[2])
    events = []
    for i in range(1, number_of_events + 1):
      try:
        print('.', end='', flush=True)
        event = Event(YEAR, f"blg-{str(i).zfill(4)}")
        points = event.points_around_peak(TIME_WINDOW)
        if len(points) > MIN_DATA_POINTS:
          events.append({'id': f"blg-{str(i).zfill(4)}", 'points': len(points), 'fbl': event.metadata['fbl'] })
      except:
        print('!', end='', flush=True)
    print()
    print('sorting...')
    events = sorted(events, key=lambda event: event['points'], reverse=True)
    print(tabulate(events))

  else:
    print('Invalid command')
    sys.exit(1)
