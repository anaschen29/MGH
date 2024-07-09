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

if not os.path.exists('Stability Experiments'):
    os.makedirs('Stability Experiments')

def single_thread_experiment_fix_n(D=40,n = 5000, step_size = 5):
    d_axis = range(1, D, step_size)
    inputs = [(d,n) for d in d_axis] 
    thread = []
    for d,n in tqdm(inputs):
        thread.append((d,experiment(d,n)))
    return thread

 
def experiment_fix_n(D=40, iterations_per_d = 20, n = 5000, step_size = 5):
  results = []
  d_axis = range(1, D, step_size)

  with ThreadPoolExecutor(max_workers = iterations_per_d) as executor:
    futures = [executor.submit(single_thread_experiment_fix_n, D, n, step_size) for _ in range(iterations_per_d)]
    results += [future.result() for future in concurrent.futures.as_completed(futures)]
  results = list(itertools.chain(*results))
#   print(results)
#   results = list(results[0])
#   print('results', results)
  new = sorted(results)
#   print(new)
  results = average_consecutive(new, iterations_per_d)
  plt.plot(d_axis, results)
  plt.xlabel("Number of features (d)")
  plt.ylabel("Volatility")
  title = "Experiment results for fixed number for data points, n="+str(n)
  plt.title(title)
  plt.savefig('Figures/fixed_n_figures/'+title+'.png')
#   plt.clear() 
  return d_axis, results 

def save_data_fixed_n(D, iterations_per_d, n, step_size, d_axis, results):
    with open('Outputs/results_fixed_n.txt', 'a') as file:
        file.write(f'Experiment D={D}, iterations_per_d = {iterations_per_d}, n = {n}, step_size = {step_size} \n')
        # file.write((D, iterations_per_d, n, step_size)+'\n')
        file.write(write_array(d_axis))
        file.write(write_array(results))



def power_law(x, a, b):
    return a * np.power(x, b)


def power_law_fit_fixed_n(d_axis, results, n, D, iterations_per_d, store = True):
    # Fit the power law to the data
    params, params_covariance = curve_fit(power_law, d_axis, results)

    # Extract the parameters
    a, b = params

    print(f"Fitted parameters: a = {a}, b = {b}")

    # Generate data points for the fitted curve
    x_fit = np.linspace(min(d_axis), max(d_axis), 100)
    y_fit = power_law(x_fit, a, b)

    # Plot the data and the fitted curve
    plt.scatter(d_axis, results, label='Data')
    plt.plot(x_fit, y_fit, color='red')
    plt.xlabel('d (number of features)')
    plt.ylabel('Volatility')
    title = f"fitted Experiment results for fixed number for data points, \n n={str(n)}, $y = {a:.2f}x^{b:.2f})$ "
    plt.title(title)
    plt.savefig('Figures/fixed_n_figures/'+title+'.png')
    if store:
        with open('Outputs/coefficients_fixed_n.txt', 'a') as file:
            file.write(f'Experiment D={D}, iterations_per_d = {iterations_per_d}, n = {n} \n')
            file.write(write_array([a, b]))

if __name__ == "__main__":
#    D, iterations_per_d, n, step_size = 100, 10, 5000, 5
#    d_axis, results = experiment_fix_n(D = 20, iterations_per_d = 10, n = 50000, step_size = 5)
#    save_data(D, iterations_per_d, n, step_size, d_axis, results)

   D, iterations_per_d, n, step_size = 50, 20, 1000, 20
   d_axis, results = experiment_fix_n(D, iterations_per_d, n, step_size)
   save_data_fixed_n(D, iterations_per_d, n, step_size, d_axis, results)
   power_law_fit_fixed_n(d_axis, results, n, D, iterations_per_d)
    