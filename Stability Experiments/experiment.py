from experiment_fixed_d import *
from experiment_fixed_n import *



if __name__ == "__main__":
    #fixed D
    #N, iterations_per_n, d, step_size
    fixed_d_inputs = [(10000, 100, 20, 20),
                      (10000, 50, 20, 20),
                      (10000, 50, 20, 20),
                      (100000, 100, 47, 20),
                      (50000, 100, 83, 15),
                      (10000 , 20, 30, 10),
                      (6236, 114, 17, 7 ),
                      (19620, 200, 147, 15),
                      (12312, 81, 100, 20),
                      (15000, 120, 25, 25),
                      (7500, 60, 22, 15),
                      (20000, 80, 30, 10),
                      (30000, 150, 50, 30),
                      (40000, 90, 40, 20),
                      (80000, 200, 70, 25),
                      (110000, 300, 95, 30),
                      (65000, 110, 55, 15),
                      (27000, 130, 60, 20),
                      (5000, 40, 18, 5),
                      (18000, 70, 28, 12),
                      (9000, 50, 21, 9),
                      (12500, 85, 33, 250),
                      (45000, 100, 45, 30),
                      (102000, 220, 120, 25)]
                    
        
    for N, step_size, d, iterations_per_n in fixed_d_inputs:
        try:
            n_axis, results = experiment_fix_d(N, iterations_per_n, d, step_size)
            save_data_fixed_d(N, iterations_per_n, d, step_size, n_axis, results)
            power_law_fit_fixed_d(n_axis, results, d, N, iterations_per_n)
        except:
            pass


    #fixed n

    # D, iterations_per_d, n, step_size

    fixed_n_inputs =   [(2000, 15, 17800, 20), #change to 2000, 15, 10000, 20
                        (1140, 10, 50000, 20),
                        (1500, 7, 47816, 15),
                        (1000 , 15, 147, 10),
                        (8191, 15, 1918346, 20),
                        (6236, 10, 6236, 114),
                        (1962, 10, 11111, 15),
                        (123, 5, 8500, 20),
                        (10, 50, 100, 5),
                        (20, 100, 200, 10),
                        (15, 60, 150, 7),
                        (30, 120, 300, 15),
                        (25, 80, 250, 12),
                        (40, 200, 400, 20),
                        (100, 25, 12500, 5),
                        (450, 10, 1250, 22),
                        (55, 20, 68900, 3),
                        (9000, 20, 120312, 25)]
    
    for D, iterations_per_d, n, step_size in fixed_n_inputs:
        try:
            d_axis, results = experiment_fix_n(D, iterations_per_d, n, step_size)
            save_data_fixed_n(D, iterations_per_d, n, step_size, d_axis, results)
            power_law_fit_fixed_n(d_axis, results, n, D, iterations_per_d)
        except:
            pass