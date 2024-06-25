import numpy as np


def bootstrap(cut_data_time, cut_data_int, cut_data_int_err, parabolic_fit, iterations):
    t_max = []
    max_values = []
    for i in range(iterations):
        indices = np.random.choice(len(cut_data_time), size=len(cut_data_time), replace=True)
        bootstrap_times = cut_data_time[indices]
        bootstrap_int = cut_data_int[indices]
        bootstrap_error = cut_data_int_err[indices]

        parabolic_fit(bootstrap_times, bootstrap_int, bootstrap_error)
        t_max.append(parabolic_fit['Tmax'].value)
        max_values.append(parabolic_fit['umin'].value)

        return max_values, t_max




