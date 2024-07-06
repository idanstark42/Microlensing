import numpy as np
from decimal import Decimal


class Value:
    def __init__(self, value, error):
        self.value = value
        self.error = error

    def __str__(self):
        rounded_error = float(f"{Decimal(f'{self.error:.2g}'):f}")
        split_error = str(rounded_error).split(".")
        decimal_places = len(split_error[1]) if len(split_error) == 2 else len(split_error[0].strip("0")) - len(
            split_error[0])
        rounded_value = round_to(self.value, decimal_places)
        return f"{rounded_value} ± {rounded_error}"

    def __repr__(self):
        return f"Value({self.value}, {self.error})"

    def n_sigma(self, other):
        return abs(self.value - other.value) / np.sqrt(self.error ** 2 + other.error ** 2)


def round_to(value, sig_figs):
    if sig_figs == 0:
        return int(value)
    return round((value * 10 ** sig_figs)) / 10 ** sig_figs


# theoretical formulas

def I(m, m_star):
    I_value = np.power(10, np.abs(m_star.value - m.value) / 2.5)
    #I_error = ((m_star.error ** 2 + m.error ** 2) ** 0.5 / 2.5) * I_value
    I_error = ((m_star.error ** 2 + m.error ** 2) ** 0.5) * ((I_value*np.log(10))/2.5)
    return Value(I_value, I_error)


def u(I, t_max):
    u_value = np.sqrt(2 * np.sqrt(I.value ** 2 / (I.value ** 2 - 1)) - 2)
    u_error = (4 * I.error / u_value) * ((I.value ** 2 + I.value - 1) / np.power(I.value ** 2 - 1, 3 / 2))
    return Value(u_value, u_error)


def u_t(t, umin, t0, tau):
    return np.sqrt(umin ** 2 + ((t - t0) / tau) ** 2)


def mu_t(t, umin, t0, tau):
    u = u_t(t, umin, t0, tau)
    return (u ** 2 + 2) / (u * np.sqrt(u ** 2 + 4))


def I_t(t, umin, t0, tau, fBL):
    mu = mu_t(t, umin, t0, tau)
    return fBL * mu + (1 - fBL)
