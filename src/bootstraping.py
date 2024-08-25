import numpy as np
from progress.bar import IncrementalBar

def bootstrap(data, callback, iterations):
  fits = []
  bar = IncrementalBar('Bootstrapping', max=iterations, suffix='%(percent)d%%')
  for i in range(iterations):
    indices = np.random.choice(len(data), size=len(data), replace=True)

    try:
      fit = callback([datum for index, datum in enumerate(data) if index in indices])
      if fit is None:
        i -= 1
      else:
        fits.append(fit)
    except Exception as e:
      print(f"Error in iteration {i}: {e}")
      i -= 1
    bar.next()
  bar.finish()
  return fits
