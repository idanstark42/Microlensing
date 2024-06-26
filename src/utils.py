import numpy as np

class Value:
  def __init__(self, value, error):
    self.value = value
    self.error = error

  def __str__(self):
    # display to 2 decimal places of error
    err = "{self.error:.2f}"
    number_of_decimal_places = len(err.split(".")[1])
    return f"{self.value:.{number_of_decimal_places}f} ± {self.error:.{number_of_decimal_places}f}"
  
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