import sys
from tabulate import tabulate

from src.regression import parabola
from src.bootstraping import bootstrap
from src.plotting import plot_data_and_parabola, plot_histogram_and_gaussians

from src.ogle import load_event

YEAR = '2019'
ID = 'blg-0001'
BOOTSTRAP_SAMPLES = 10000
MIN_DATA_POINTS = 30
TIME_WINDOW = 20

def points_around_peak(data):
  max_magnitude = max([datum['m'].value for datum in data])
  peak_time = [datum['t'] for datum in data if datum['m'].value == max_magnitude][0]
  return [datum for datum in data if abs(datum['t'] - peak_time) < TIME_WINDOW / 2]

def part_1():
  event = load_event(YEAR, ID)

  cut_data = points_around_peak(event.data, 10)

  if (len(cut_data) < MIN_DATA_POINTS):
    print(f"Data has only {len(cut_data)} points, which is less than the minimum of {MIN_DATA_POINTS}")
    sys.exit(1)

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

  elif command == 'event':
    print(load_event(YEAR, ID))

  elif command == 'max_points':
    event = load_event(YEAR, ID)
    points = points_around_peak(event.data)
    print(f"Number of points around peak: {len(points)}")
    tabulated_points = [[point['t'], point['m']] for point in points]
    print(tabulate(tabulated_points, headers=['Time', 'Magnitude']))

  elif command == 'good_events':
    number_of_events = int(sys.argv[2])
    events = []
    for i in range(1, number_of_events + 1):
      try:
        print('.', end='', flush=True)
        event = load_event(YEAR, f'blg-{str(i).zfill(4)}')
        points = points_around_peak(event.data)
        if len(points) > MIN_DATA_POINTS:
          events.append({ 'id': f'blg-{str(i).zfill(4)}', 'points': len(points) })
      except:
        print('!', end='', flush=True)
        pass
    print()
    print('sorting...')
    events = sorted(events, key=lambda event: event['points'], reverse=True)
    print(tabulate(events))

  else:
    print('Invalid command')
    sys.exit(1)
