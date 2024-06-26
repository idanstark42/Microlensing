import numpy as np

class Value:
  def __init__(self, value, error):
    self.value = value
    self.error = error

  def __str__(self):
    # display to last 2 significant figures of error
    if ('.' not in str(self.error)):
      return f"{self.value} ± {self.error}"
    if (self.error == 0):
      return str(self.value)
    round_sig = lambda x, sig: round(x, sig - int(np.floor(np.log10(abs(x)))) - 1)
    error_str = '{:.2e}'.format(self.error)
    significant_digits = len(error_str.split('e')[0].replace('.', '').rstrip('0'))
    rounded_error = round_sig(self.error, significant_digits)
    rounded_value = round(self.value, significant_digits - int(np.floor(np.log10(abs(self.error)))) - 1)
    return f"{rounded_value} ± {rounded_error}"
  
  def __repr__(self):
    return f"Value({self.value}, {self.error})"
  
  def n_sigma (self, other):
    return (self.value - other.value) / np.sqrt(self.error ** 2 + other.error ** 2)

# theoretical formulas

def I(m, m_star):
  I_value = np.power(10, np.abs(m_star.value - m.value) / 2.5)
  I_error = ((m_star.error ** 2 + m.error ** 2) ** 0.5 / 2.5) * I_value
  return Value(I_value, I_error)

def u(I):
  u_value = np.sqrt(2 * np.sqrt(I.value ** 2 / (I.value ** 2 - 1)) - 2)
  u_error = (4 * I.error / u_value) * ((I.value ** 2 + I.value - 1) / np.power(I.value ** 2 - 1, 3 / 2))
  return Value(u_value, u_error)