import numpy as np
import random 
import time
import os 
import pickle 


def optimizer(X, Y):
    return np.linalg.inv(X.T@X) @ (X.T@Y)

# def new_distribution(d,n, covariance_matrix = None):
#     np.random.seed(int(time.time())) #set seed 

#     if not covariance_matrix: #draw distribution
#         S = np.random.normal(loc=0, scale=10, size=(n, d))

#     raise NotImplementedError


def distribution(d, n):
    mean = 0
    variance = 0.2
    c = np.random.randn(d)
    c = c.reshape(1,d)
    x = np.random.normal(loc=0, scale=10, size=(n, d))
    out = c@x.T + np.random.normal(loc=mean, scale=np.sqrt(variance), size=n)
    return  x, out.T, c

def write_array(array):
    return' '.join(map(str, array))+'\n' # Join elements with a space delimiter

def average_consecutive(array, size):
    output = []
    for i in range(0, len(array), size):
        output.append(np.average(array[i:i+size]))
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

def distribution(Sigma, n):
    d = Sigma.shape[0]
    mean = 0
    variance = 0.2
    c = np.random.randn(d)
    c = c.reshape(1,d)
    x = draw_multivariate_normal(Sigma, n)
    out = c@x.T + np.random.normal(loc=mean, scale=np.sqrt(variance), size=n)
    return  x, out.T, c

def experiment(Sigma, n): 
  np.random.seed(int(time.time()))

  while True:
    try:
      X1, Y1, _ = new_distribution(Sigma, n)
      X2, Y2, _ = new_distribution(Sigma, n)
      X3 = draw_multivariate_normal(Sigma, 1)

      left = (optimizer(X1, Y1) - optimizer(X2, Y2)).T
      return ((left @ (X3.T)).item())**2
    except:
       print('failed')
       pass

    

Sigma = np.array([[1, 0.5], [0.5, 2]])  # 2x2 covariance matrix
num_samples = 1000
print(experiment(Sigma, num_samples))