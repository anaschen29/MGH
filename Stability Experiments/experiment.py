from experiment_fixed_d import *
from experiment_fixed_n import *



if __name__ == "__main__":

    #N, step_size, d, iterations_per_n
    fixed_d_inputs = [(1000, 100, 25, 10)]
    # fixed_d_inputs = [(10000, 100, 20, 20),
    #                   (100000, 500, 47, 20),
    #                   (50000, 100, 83, 15),
    #                   (10000 , 20, 30, 10),
    #                   (6236, 114, 17, 7 ),
    #                   (19620, 200, 147, 15),
    #                   (12312, 81, 100, 20)]
    
    for N, step_size, d, iterations_per_n in fixed_d_inputs:
        n_axis, results = experiment_fix_d(N, iterations_per_n, d, step_size)
        save_data(N, iterations_per_n, d, step_size, n_axis, results)
        power_law_fit_fixed_d(n_axis, results, d, N, iterations_per_n)

    # D, iterations_per_d, n, step_size
    fixed_n_inputs = [(2000, 15, 10000, 20),
                      (1140, 10, 50000, 20),
                      (1500, 7, 47816, 15),
                      (1000 , 15, 147, 10),
                      (6236, 10, 6236, 114),
                      (1962, 10, 11111, 15),
                      (123, 5, 8500, 20)]
    # fixed_n_inputs = [(100, 5, 875, 10)]
    # for D, iterations_per_d, n, step_size in fixed_n_inputs:
    #     d_axis, results = experiment_fix_d(D, iterations_per_d, n, step_size)
    #     save_data(D, iterations_per_d, n, step_size, d_axis, results)
    #     power_law_fit_fixed_n(d_axis, results, n, D, iterations_per_d)