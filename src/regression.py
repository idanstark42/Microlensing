from iminuit import Minuit
import numpy as np


def parabolic_fit(cut_data_time, cut_data_int, cut_data_int_err):
    def parabola(a2, a1, a0, cut_data_time):
        return a2 * (cut_data_time ** 2) + a1 * cut_data_time + a0

    def likelihood(a2, a1, a0):
        model = parabola(a2, a1, a0, cut_data_time)
        return np.sum(((cut_data_int - model) / cut_data_int_err) ** 2)

    initial_params = {'a2': -10, 'a1': 10, 'a0': 10}
    fit = Minuit(likelihood, **initial_params)
    fit.migrad()
    t_max = -fit[1]/(2*fit[0])
    u_min = parabola(fit[0], fit[1], fit[2], t_max)
    return dict(a2=fit[0], a1=fit[1], a0=fit[2], chi2=fit.fval, Tmax=t_max, umin=u_min)
