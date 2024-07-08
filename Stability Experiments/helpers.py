import numpy as np
import random 
import time


def optimizer(X, Y):
    return np.linalg.inv(X.T@X) @ (X.T@Y)


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



def experiment(d,n):
  np.random.seed(int(time.time()))

  while True:
    try:
      X1, Y1, c1 = distribution(d, n)
      X2, Y2, c2 = distribution(d, n)
      X3, _, _ = distribution(d, n)
      return np.linalg.norm((optimizer(X1, Y1) - optimizer(X2, Y2)).T @ X3.T)
    except:
      print('failed')
      pass