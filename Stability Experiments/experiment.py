import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import tqdm
import itertools
from tqdm import tqdm
from collections import defaultdict
from scipy.optimize import curve_fit

import csv
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed

# plt.ioff()
# plt.use('Agg')

def distribution(d, n):
    mean = 0
    variance = 0.2
    c = np.random.randn(d)
    c = c.reshape(1,d)
    x = np.random.normal(loc=0, scale=10, size=(n, d))
    out = c@x.T + np.random.normal(loc=mean, scale=np.sqrt(variance), size=n)
    return  x, out.T, c


def optimizer(X, Y):
    return np.linalg.inv(X.T@X) @ (X.T@Y)


def experiment(d,n):
    while True:
        try:
            X1, Y1, c1 = distribution(d, n)
            X2, Y2, c2 = distribution(d, n)
            X3, _, _ = distribution(d, n)
            X1, Y1, c1 = distribution(d, n)
            X2, Y2, c2 = distribution(d, n)
            X3, _, _ = distribution(d, n)

            return np.linalg.norm((optimizer(X1, Y1) - optimizer(X2, Y2)).T @ X3.T)
        except:
            print('failed')
            pass


def average_consecutive(array, size):
    output = []
    for i in range(0, len(array), size):
        output.append(np.average(array[i:i+size]))
    return output


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
  plt.savefig('Figures/'+title+'.png')
#   plt.clear() 
  return d_axis, results 

def write_array(array):
    return' '.join(map(str, array))+'\n' # Join elements with a space delimiter



def save_data(D, iterations_per_d, n, step_size, d_axis, results):
    with open('Outputs/results.txt', 'a') as file:
        file.write(f'Experiment D={D}, iterations_per_d = {iterations_per_d}, n = {n}, step_size = {step_size} \n')
        # file.write((D, iterations_per_d, n, step_size)+'\n')
        file.write(write_array(d_axis))
        file.write(write_array(results))



def power_law(x, a, b):
    return a * np.power(x, b)


def power_law_fit(d_axis, results):
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
    plt.plot(x_fit, y_fit, label=f'Fitted curve: $y = {a:.2f}x^{b:.2f})$', color='red')
    plt.xlabel('d (number of features)')
    plt.ylabel('Volatility')
    title = "fitted Experiment results for fixed number for data points, n="+str(n)
    plt.title(title)
    plt.savefig('Figures/'+title+'.png')

if __name__ == "__main__":
#    D, iterations_per_d, n, step_size = 100, 10, 5000, 5
#    d_axis, results = experiment_fix_n(D = 20, iterations_per_d = 10, n = 50000, step_size = 5)
#    save_data(D, iterations_per_d, n, step_size, d_axis, results)

   D, iterations_per_d, n, step_size = 1000, 20, 10000, 20
   d_axis, results = experiment_fix_n(D, iterations_per_d, n, step_size)
   save_data(D, iterations_per_d, n, step_size, d_axis, results)
   power_law_fit(d_axis, results)


