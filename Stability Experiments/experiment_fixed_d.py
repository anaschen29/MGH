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
import sys

curr_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(curr_dir)
output_folder = os.path.join(curr_dir, "Figures/fixed_d_figures/")
# print(type(output_folder))

def new_single_thread_experiment_fix_d(Sigma, c, lam = 0, N = 40, step_size = 50, num_points = 15):
    d, _ = np.shape(Sigma)
    n_axis = range(3*d, N, step_size)
    inputs = [(Sigma, n) for n in n_axis]
    thread = []
    for Sigma, n in tqdm(inputs):
        thread.append((n, experiment(Sigma, n, c, lam, num_points)))
    return thread

default_d = 29
 
def new_experiment_fix_d(lam = 0, c = np.random.randn(default_d).reshape(1,default_d),N = 100, iterations_per_n = 20, Sigma = generate_random_covariance_matrix(default_d), step_size = 50):
    results = []
    d, _ = Sigma.shape
    n_axis = range(3*d, N, step_size)
    with ThreadPoolExecutor(max_workers=iterations_per_n) as executor:
        futures = [executor.submit(new_single_thread_experiment_fix_d, Sigma, c, lam, N, step_size) for _ in range(iterations_per_n)]
        results += [future.result() for future in concurrent.futures.as_completed(futures)]
    print(results)
    results = list(itertools.chain(*results))
    new = sorted(results)
    results = new_average_consecutive(new)
    plt.plot(n_axis, results)
    plt.xlabel("Number of samples (n)")
    plt.ylabel("Volatility")
    title = f"Volatility for fixed d = {str(d)}, lambda = {lam}" 
    plt.title(title)
    plt.savefig(output_folder+title+'.png')
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
  # pass
  d = 21
  c = np.random.randn(d).reshape(1, d)
  Sigma = generate_random_covariance_matrix(d)

  N = 10000
  iterations_per_n = 40
  step_size = 50
  for lam in [0.0, 0.2, 0.4]:
    n_axis, results = new_experiment_fix_d(lam, c, N, iterations_per_n, Sigma = Sigma, step_size = step_size)
  
  