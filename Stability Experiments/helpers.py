import numpy as np
import random 
import time
import os 
import pickle 
import concurrent
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def optimizer(X, Y, lam = 0):
    d = np.shape(X)[1]
    return np.linalg.inv((X.T@X) + lam * np.eye(d)) @ (X.T@Y)

# def new_distribution(d,n, covariance_matrix = None):
#     np.random.seed(int(time.time())) #set seed 

#     if not covariance_matrix: #draw distribution
#         S = np.random.normal(loc=0, scale=10, size=(n, d))

#     raise NotImplementedError


def write_array(array):
    return' '.join(map(str, array))+'\n' # Join elements with a space delimiter

def average_consecutive(array, size):
    output = []
    for i in range(0, len(array), size):
        output.append(np.average(array[i:i+size]))
    return output

def new_average_consecutive(array):
    table = {}
    for n,v in array:
      try:
          table[n].append(v)
      except:
          table[n] = []

    output = []
    for n in sorted(table):
       output.append(np.average(table[n]))
    
    return output

                
#claude AI
def draw_multivariate_normal(Sigma, n = 1):
  d = Sigma.shape[0]
  L = np.linalg.cholesky(Sigma)
  # Generate a vector of independent standard normal random variables
  Z = np.random.standard_normal((d, n))
  # Transform z to get the desired distribution
  X = np.dot(L, Z).T
  
  return X #of shape nxd

def distribution(Sigma, n, c, variance = 0.2):
    d = Sigma.shape[0]
    mean = 0
    x = draw_multivariate_normal(Sigma, n)
    out = c@x.T + np.random.normal(loc=mean, scale=np.sqrt(variance), size=n)
    return  x, out.T, c

def experiment(Sigma, n, c, lam = 0, num_points = 1): 
  np.random.seed(int(time.time()))
  while True:
    try:
      X1, Y1, _ = distribution(Sigma, n, c)
      X2, Y2, _ = distribution(Sigma, n, c)
      X3 = draw_multivariate_normal(Sigma, num_points)

      left = (optimizer(X1, Y1, lam) - optimizer(X2, Y2, lam)).T
      return np.linalg.norm((left @ (X3.T)))**2
    except:
       print('failed')
       pass


def generate_random_covariance_matrix(d, error = 1e-6):
    # Generate a random matrix A
    while True:
      A = np.random.rand(d, d)
      if abs(np.linalg.det(A)) < error:
         continue
    # Compute A * A^T to ensure positive semi-definiteness
      cov = np.dot(A, A.T)
    
      return cov



def single_thread_covariance_expectation_approximator(Sigma, n, rounds = 10000, timer = False):
    start = time.time()
    values = 0
    for _ in tqdm(range(rounds)):
      # while True: 
        # try:
          X1 = draw_multivariate_normal(Sigma, n)
          X2 = draw_multivariate_normal(Sigma, n)
          X = draw_multivariate_normal(Sigma, 1)
          S1 = (X1.T @ X1)
          S2 =  (X2.T @ X2)
          # print(np.shape(X), np.shape(S1), np.shape(X1), np.shape(X2), np.shape(S2), np.shape(X))
          values += (X @ np.linalg.inv(S1) @ (X1.T @ X2) @ np.linalg.inv(S2) @ X.T)
          # break
        # except:
          # pass
    
    end = time.time()
    if timer:
      print("single thread analogue took", end-start)
    return values/rounds

def multi_thread_covariance_expectation_approximator(Sigma, n, rounds):
    start = time.time()
    results = []
    d, _ = Sigma.shape
    with ThreadPoolExecutor(max_workers = 10) as executor:
        futures = [executor.submit(single_thread_covariance_expectation_approximator, Sigma, n, rounds//10) for _ in range(10)]
        results += [future.result() for future in concurrent.futures.as_completed(futures)]
    end = time.time()
    print("multi thread analogue took", end-start)
    return np.mean(results)


def yury_experiment(Sigma, n, rounds = 10000):
    values = 0
    for _ in tqdm(range(rounds)):
      while True: 
        try:
          X1 = draw_multivariate_normal(Sigma, n)
          # X2 = draw_multivariate_normal(Sigma, n)
          S1 = 1/n * (X1.T @ X1)
          # S2 = 1/n * (X2.T @ X2)
          # S12 = 1/n * (X1.T @ X2)
          values += np.trace(Sigma @ np.linalg.inv(S1))
          break
        except:
          pass
    return values/rounds


if __name__ == "__main__":
  
  d = 47
  n = 18857
  Sigma = generate_random_covariance_matrix(d)
  c = np.random.randn(d).reshape(1, d)
  print(experiment(Sigma, n, c, lam = 0.1, num_points = 12))
  # print(yury_experiment(Sigma, 1231))

  # print(multi_thread_covariance_expectation_approximator(Sigma, n, rounds))
  # print(single_thread_covariance_expectation_approximator(Sigma, n, rounds))
  # print(single_thread_covariance_expectation_approximator(Sigma, 2*n, rounds))
  # print(single_thread_covariance_expectation_approximator(Sigma, 4*n, rounds))