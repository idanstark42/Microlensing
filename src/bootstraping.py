import numpy as np


def bootstrap(cut_data_time, cut_data_int, cut_data_int_err, I0, parabolic_fit, iterations):
    t_max = []
    max_values = []
    taus = []
    for i in range(iterations):
        indices = np.random.choice(len(cut_data_time), size=len(cut_data_time), replace=True)
        bootstrap_times = cut_data_time[indices]
        bootstrap_int = cut_data_int[indices]
        bootstrap_error = cut_data_int_err[indices]

        fit = parabolic_fit(bootstrap_times, bootstrap_int, bootstrap_error, I0)
        taus.append(fit['tau'])
        t_max.append(fit['Tmax'])
        max_values.append(fit['umin'])

    return max_values, t_max, taus
