import sys

import matplotlib.pyplot as plt
from tabulate import tabulate
from src.regression import fit_polynomial, fit_histogram_gaussian, gaussian, generate_chi_squared_nd_map
from src.bootstraping import bootstrap
from src.plotting import plot_chi_squared_map_gridmap, plot_chi_squared_map_contour, plot_event, plot_data_and_parabola, \
    plot_histogram_and_gaussian, plot_full_fit, corner_plot
import numpy as np
from src.ogle import Event
from src.utils import Value, I, I_t
from src.settings import YEAR, ID, BOOTSTRAP_SAMPLES, MIN_DATA_POINTS, TIME_WINDOW


def part_1(graphs=True):
    print()
    print('--- part 1 ---')
    event = Event(YEAR, ID)
    data = event.points_around_peak(TIME_WINDOW)

    if len(data) < MIN_DATA_POINTS:
        print(f'Data has only {len(data)} points, which is less than the minimum of {MIN_DATA_POINTS}. Exiting.')
        sys.exit(1)

    print(f'Found {len(data)} points around the peak. Continuing.')
    print()
    print('1. Fitting parabola with all points...')
    parabola_prediction = fit_polynomial(data)
    print('2. Bootstrapping...')
    bootstrap_predictions = bootstrap(data, fit_polynomial, BOOTSTRAP_SAMPLES)

    FIELDS = ['tau', 'umin', 'Tmax']

    gaussians = {field: fit_histogram_gaussian([prediction[field].value for prediction in bootstrap_predictions]) for
                 field in FIELDS}
    gaussian_predictions = {key: Value(gaussians[key][1], gaussians[key][2]) for key in gaussians}

    print('3. Done')
    print()
    print(f"χ²:\t{parabola_prediction['chi2']}")
    print(f"a0: {parabola_prediction['a0']}\ta1: {parabola_prediction['a1']}\ta2: {parabola_prediction['a2']}")
    print(tabulate([[
        field,
        event.metadata[field],
        parabola_prediction[field],
        gaussian_predictions[field],
        parabola_prediction[field].n_sigma(event.metadata[field]),
        abs(parabola_prediction[field].value - gaussian_predictions[field].value) / gaussian_predictions[field].value,
        gaussians[field][3]
    ] for field in FIELDS],
        headers=['Parameter', 'OGLE', 'Parabola', 'Histogram', 'Nsigma', 'Difference', 'Gaussian χ²']))

    if not graphs:
        return

    plot_data_and_parabola(data, parabola_prediction)
    for field in FIELDS:
        plot_histogram_and_gaussian([parabola_prediction[field].value for parabola_prediction in bootstrap_predictions],
                                    field, lambda x: gaussian(x, *gaussians[field][:3]))


def part_2(graphs=True):
    print()
    print('--- part 2 ---')
    event = Event(YEAR, ID)

    print('expected values: Tmax:', event.metadata['Tmax'], 'umin:', event.metadata['umin'])
    print('1. Loading parabolic fit...')
    parabola_prediction = fit_polynomial(event.points_around_peak(TIME_WINDOW))

    print('2. Generating chi squared map...')
    data = event.data
    umin_p, tmax_p = parabola_prediction['umin'].value, parabola_prediction['Tmax'].value
    get_fit = lambda t, parameters: I_t(t, parameters['umin'], parameters['Tmax'], event.metadata['tau'].value,
                                        event.metadata['fbl'].value, event.metadata['I*'].value)
    dimensions = {"umin": (umin_p, 0.6, 100), "Tmax": (tmax_p, 50, 200)}
    chi2_map, best_params = generate_chi_squared_nd_map(dimensions, data, get_fit, 2)

    print('3. Bootstrapping...')
    bootstrap_predictions = bootstrap(data, lambda data: generate_chi_squared_nd_map(dimensions, data, get_fit, 2)[1],
                                      BOOTSTRAP_SAMPLES)

    FIELDS = ['umin', 'Tmax']

    gaussians = {field: fit_histogram_gaussian([prediction[field].value for prediction in bootstrap_predictions]) for
                 field in FIELDS}
    gaussian_predictions = {key: Value(gaussians[key][1], gaussians[key][2]) for key in gaussians}

    print('4. Done')

    print(tabulate([[
        key,
        event.metadata[key],
        best_params[key],
        parabola_prediction[key],
        gaussian_predictions[key],
        abs(best_params[key].value - event.metadata[key].value) / event.metadata[key].value
    ] for key in FIELDS],
        headers=['Parameter', 'OGLE', 'Best', 'Parabola', 'Bootstrapping', 'Difference']))

    if not graphs:
        return

    plot_full_fit(data, lambda t: get_fit(t, {
        key: best_params[key].value if type(best_params[key]) == Value else best_params[key] for key in best_params}))
    plot_chi_squared_map_gridmap(chi2_map, dimensions, {})
    plot_chi_squared_map_contour(chi2_map, dimensions, {})


def part_3(graphs=True):
    print()
    print('--- part 3 ---')
    event = Event(YEAR, ID)
    data = event.data
    umin, Tmax, fbl, tau, I_min = event.metadata['umin'].value, event.metadata['Tmax'].value, event.metadata[
        'fbl'].value, event.metadata['tau'].value, event.metadata['I*'].value
    dimensions = {
        "umin": (umin, 1.5, 20),
        "Tmax": (Tmax, 1.5, 20),
        "fbl": (fbl, 1.5, 20),
        "tau": (tau, 1.5, 20)
    }

    print('expected values: Tmax:', event.metadata['Tmax'], 'umin:', event.metadata['umin'], 'tau:',
          event.metadata['tau'], 'fbl:', event.metadata['fbl'], )
    print('1. Generating chi squared map...')
    get_fit = lambda t, parameters: I_t(t, parameters['umin'], parameters['Tmax'], parameters['tau'], parameters['fbl'],
                                        I_min)
    chi2_map, best_params = generate_chi_squared_nd_map(dimensions, data, get_fit, 4)

    print('2. Done')

    print(tabulate([[key, event.metadata[key], best_params[key]] for key in dimensions.keys()],
                   headers=['Parameter', 'OGLE', 'Best']))

    if not graphs:
        return

    plot_full_fit(data, lambda t: get_fit(t, {
        key: best_params[key].value if type(best_params[key]) == Value else best_params[key] for key in best_params}))
    corner_plot(chi2_map, dimensions)


if __name__ == '__main__':
    command = sys.argv[1]

    if command == 'part1':
        part_1('--no-graphs' not in sys.argv[2:] if len(sys.argv) > 2 else True)

    elif command == 'part2':
        part_2('--no-graphs' not in sys.argv[2:] if len(sys.argv) > 2 else True)

    elif command == 'part3':
        part_3('--no-graphs' not in sys.argv[2:] if len(sys.argv) > 2 else True)

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
                    events.append({'id': f"blg-{str(i).zfill(4)}", 'points': len(points), 'fbl': event.metadata['fbl']})
            except:
                print('!', end='', flush=True)
        print()
        print('sorting...')
        events = sorted(events, key=lambda event: event['points'], reverse=True)
        print(tabulate(events, headers='keys'))

    else:
        print('Invalid command')
        sys.exit(1)
