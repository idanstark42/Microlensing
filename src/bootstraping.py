import numpy as np
from progress.bar import IncrementalBar

def bootstrap(data, callback, iterations):
  fits = []
  bar = IncrementalBar('Bootstrapping', max=iterations, suffix='%(percent)d%%')
  for i in range(iterations):
    indices = np.random.choice(len(data), size=len(data), replace=True)

    fit = callback([datum for index, datum in enumerate(data) if index in indices])
    fits.append(fit)
    bar.next()
  bar.finish()
  return fits
