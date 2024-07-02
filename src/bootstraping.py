import numpy as np


def bootstrap(data, callback, iterations):
    fits = []
    for i in range(iterations):
        indices = np.random.choice(len(data), size=len(data), replace=True)

        fit = callback([datum for index, datum in enumerate(data) if index in indices])
        fits.append(fit)

    return fits
