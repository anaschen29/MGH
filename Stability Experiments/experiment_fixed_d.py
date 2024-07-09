import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
import itertools
from tqdm import tqdm
from collections import defaultdict
from scipy.optimize import curve_fit
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from helpers import *
import os

import os



#fix the starting point of the N range to be 3*d
def single_thread_experiment_fix_d(N = 40, d = 50, step_size = 50):
    n_axis = range(3*d, N, step_size)
    inputs = [(d,n) for n in n_axis] 
    thread = []
    for d,n in tqdm(inputs):
        thread.append((n,experiment(d,n)))
    return thread

 
def experiment_fix_d(N = 100, iterations_per_n = 20, d = 50, step_size = 50):
  results = []
  n_axis = range(3*d, N, step_size)

  with ThreadPoolExecutor(max_workers = iterations_per_n) as executor:
    futures = [executor.submit(single_thread_experiment_fix_d, N, d, step_size) for _ in range(iterations_per_n)]
    results += [future.result() for future in concurrent.futures.as_completed(futures)]
  results = list(itertools.chain(*results))
#   print(results)
#   results = list(results[0])
#   print('results', results)
  new = sorted(results)
#   print(new)
  results = average_consecutive(new, iterations_per_n)
  plt.plot(n_axis, results)
  plt.xlabel("Number of samples (n)")
  plt.ylabel("Volatility")
  title = "Experiment results for fixed number of features, d="+str(d)
  plt.title(title)
  plt.savefig('Figures/fixed_d_figures/'+title+'.png')
#   plt.clear() 
  return n_axis, results 


def save_data_fixed_d(N, iterations_per_n, d, step_size, n_axis, results):
    with open('Outputs/results_fixed_d.txt', 'a') as file:
        file.write(f'Experiment N={N}, iterations_per_n = {iterations_per_n}, d = {d}, step_size = {step_size} \n')
        file.write(write_array(n_axis))
        file.write(write_array(results))



def power_law(x, a, b):
    return a * np.power(x, b)


def power_law_fit_fixed_d(n_axis, results, d, N, iterations_per_n, store = True):
    # Fit the power law to the data
    params, params_covariance = curve_fit(power_law, n_axis, results)

    # Extract the parameters
    a, b = params

    print(f"Fitted parameters: a = {a}, b = {b}")

    # Generate data points for the fitted curve
    x_fit = np.linspace(min(n_axis), max(n_axis), 100)
    y_fit = power_law(x_fit, a, b)

    # Plot the data and the fitted curve
    plt.scatter(n_axis, results, label='Data')
    plt.plot(x_fit, y_fit, color='red')
    plt.xlabel('n (number of samples)')
    plt.ylabel('Volatility')
    title = f"fitted Experiment results for fixed number of features, \n d={str(d)}, $y = {a:.2f}x^{b:.2f})$ "
    plt.title(title)
    plt.savefig('Figures/fixed_d_figures/'+"fitted Experiment results for fixed number of features, \n d={str(d)}.png")
    if store:
        with open('Outputs/coefficients_fixed_d.txt', 'a') as file:
            file.write(f'Experiment N={N}, iterations_per_n = {iterations_per_n}, d = {d} \n')
            file.write(write_array([a, b]))
if __name__ == "__main__":
   N = 20000
   step_size = 500
   d = 47
   iterations_per_n = 15

   n_axis, results = experiment_fix_d(N, iterations_per_n, d, step_size)
   save_data_fixed_d(N, iterations_per_n, d, step_size, n_axis, results)
   power_law_fit_fixed_d(n_axis, results, d, N, iterations_per_n)
    # print(experiment(20, 1000))


