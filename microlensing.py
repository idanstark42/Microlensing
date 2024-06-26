import sys
from tabulate import tabulate
import numpy as np
from src.regression import parabolic_fit, tau_from_parabola
from src.bootstraping import bootstrap
from src.plotting import plot_data_and_parabola, plot_histogram_and_gaussians

from src.ogle import load_event

YEAR = '2019'
ID = 'blg-0171'
BOOTSTRAP_SAMPLES = 10000
MIN_DATA_POINTS = 20
TIME_WINDOW = 7


def points_around_peak(event):
    return [datum for datum in event.data if abs(datum['t'] - event.metadata['Tmax'].value) < TIME_WINDOW / 2]


def n_sigma(value1, value1_err, comparance):
    return np.abs(value1 - comparance.value) / (value1_err ** 2 + comparance.error ** 2) ** 0.5


def part_1():
    event = load_event(YEAR, ID)
    #cut_data = points_around_peak(event)
    cut_data = event.data
    cut_data_time = np.array([cut_data[i]['t'] for i in range(len(cut_data))])
    cut_data_int = np.array([cut_data[i]['m'].value for i in range(len(cut_data))])
    cut_data_int_err = np.array([np.float64(cut_data[i]['m'].error) for i in range(len(cut_data))])

    if len(cut_data) < MIN_DATA_POINTS:
        print(f"Data has only {len(cut_data)} points, which is less than the minimum of {MIN_DATA_POINTS}")
        sys.exit(1)

    predication = parabolic_fit(cut_data_time, cut_data_int, cut_data_int_err, event.metadata['I0'])
    print(f"Chi2 Value for parabolic fit: {predication['chi2']}")
    plot_data_and_parabola(cut_data_time, cut_data_int, cut_data_int_err, predication)

    tau_n_sigma = n_sigma(predication['tau'], predication['tau_err'], event.metadata['tau'])
    umin_n_sigma = n_sigma(predication['umin'], predication['umin_err'], event.metadata['umin'])
    tmax_n_sigma = n_sigma(predication['Tmax'], predication['Tmax_err'], event.metadata['Tmax'])

    print(f"{tau_n_sigma}   {tmax_n_sigma}")
    print('Finished first fit')
    print(tabulate([
        ['tau', predication['tau'], event.metadata['tau'], tau_n_sigma],
        ['umin', predication['umin'], event.metadata['umin'], umin_n_sigma],
        ['Tmax', predication['Tmax'], event.metadata['Tmax'], tmax_n_sigma]
    ], headers=['Parameter', 'Fit', 'OGLE', 'N Sigma']))

    bootstrap_umin, bootstrap_tmax, bootstrap_tau = bootstrap(cut_data_time, cut_data_int, cut_data_int_err, event.metadata['I0'],
                                                              lambda data_t, data_int, data_int_err, I0: parabolic_fit(
                                                                  data_t, data_int,data_int_err, I0), BOOTSTRAP_SAMPLES)

    plot_histogram_and_gaussians(bootstrap_tau, predication['tau'], event.metadata['tau'], 'tau')
    plot_histogram_and_gaussians(bootstrap_umin, predication['umin'], predication['umin_err'], 'umin')
    plot_histogram_and_gaussians(bootstrap_tmax, predication['Tmax'], predication['Tmax_err'], 'Tmax')


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
        points = points_around_peak(event)
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
                points = points_around_peak(event)
                if len(points) > MIN_DATA_POINTS:
                    events.append({'id': f'blg-{str(i).zfill(4)}', 'points': len(points)})
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
