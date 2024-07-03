import sys
from tabulate import tabulate
import numpy as np
from src.regression import fit_polynomial
from src.bootstraping import bootstrap
from src.plotting import plot_event, plot_data_and_parabola, plot_histogram_and_gaussians
from src.ogle import Event
from src.utils import Value, I

YEAR = '2019'
ID = 'blg-0027'
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

  print()
  print('1. Fitting parabola with all points...')
  parabola_prediction = fit_polynomial(data)
  print('2. Bootstrapping...')
  bootstrap_predictions = bootstrap(data, fit_polynomial, BOOTSTRAP_SAMPLES)

  histogram_predictions = {
    'tau': Value(np.mean([parabola_prediction['tau'].value for parabola_prediction in bootstrap_predictions]), np.std([parabola_prediction['tau'].value for parabola_prediction in bootstrap_predictions])),
    'umin': Value(np.mean([parabola_prediction['umin'].value for parabola_prediction in bootstrap_predictions]), np.std([parabola_prediction['umin'].value for parabola_prediction in bootstrap_predictions])),
    'Tmax': Value(np.mean([parabola_prediction['Tmax'].value for parabola_prediction in bootstrap_predictions]), np.std([parabola_prediction['Tmax'].value for parabola_prediction in bootstrap_predictions]))
  }

  print()
  print('Done')
  print(f"χ²:\t{parabola_prediction['chi2']}")
  print(tabulate([
    ['tau', event.metadata['tau'], parabola_prediction['tau'], histogram_predictions['tau'], parabola_prediction['tau'].n_sigma(event.metadata['tau']), abs(parabola_prediction['tau'].value - histogram_predictions['tau'].value) / histogram_predictions['tau'].value],
    ['umin', event.metadata['umin'], parabola_prediction['umin'], histogram_predictions['umin'], parabola_prediction['umin'].n_sigma(event.metadata['umin']), abs(parabola_prediction['umin'].value - histogram_predictions['umin'].value) / histogram_predictions['umin'].value],
    ['Tmax', event.metadata['Tmax'], parabola_prediction['Tmax'], histogram_predictions['Tmax'], parabola_prediction['Tmax'].n_sigma(event.metadata['Tmax']), abs(parabola_prediction['Tmax'].value - histogram_predictions['Tmax'].value) / histogram_predictions['Tmax'].value]
  ], headers=['Parameter', 'OGLE', 'Parabola', 'Histogram', 'Nsigma', 'Difference']))

  plot_data_and_parabola(data, parabola_prediction)
  plot_histogram_and_gaussians([parabola_prediction['tau'].value for parabola_prediction in bootstrap_predictions], 'tau')
  plot_histogram_and_gaussians([parabola_prediction['umin'].value for parabola_prediction in bootstrap_predictions], 'umin')
  plot_histogram_and_gaussians([parabola_prediction['Tmax'].value for parabola_prediction in bootstrap_predictions], 'Tmax')

def part_2():
  print('--- calulating non-linear fit ---')
  event = Event(YEAR, ID)
  data = event.points_around_peak(TIME_WINDOW)

  if len(data) < MIN_DATA_POINTS:
    print(f'Data has only {len(data)} points, which is less than the minimum of {MIN_DATA_POINTS}. Exiting.')
    sys.exit(1)

  print()
  print('1. Fitting parabola with all points...')
  parabola_prediction = fit_polynomial(data)

  print('2. calculating non-linear fit')

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
    print(tabulate(events, headers='keys'))

  else:
    print('Invalid command')
    sys.exit(1)
