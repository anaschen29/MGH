import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm
from tqdm import tqdm
from collections import defaultdict
from scipy.optimize import curve_fit
from helpers import *
import multiprocessing
import os
import torch
from model_classes import LinearRegressor
import torch.nn as nn
import torch.optim as optim
import warnings

warnings.filterwarnings("ignore", category = UserWarning)
# helper for both

## draw_data(D, n, with_replacement = True)

def draw_data(X, y, n, with_replacement = True):
    num_samples = X.shape[0]
    indices = np.random.choice(num_samples, size = n, replace = True)
    return X[indices], y[indices]

def generate_train_test_data(X, y, n, alpha: float = 0, test_size = 100):
    intersectionX, intersectionY = draw_data(X, y, int(alpha*n))
    rest1X, rest1y = draw_data(X, y, n - int(alpha*n))
    rest2X, rest2y = draw_data(X, y, n - int(alpha*n))
    testX, _ = draw_data(X, y, test_size)

    S1_X = np.concatenate((intersectionX, rest1X))
    S2_X = np.concatenate((intersectionX, rest2X))
    S1_y = np.concatenate((intersectionY, rest1y))
    S2_y = np.concatenate((intersectionY, rest2y))

    S1 = (S1_X, S1_y)
    S2 = (S2_X, S2_y)
    
    return S1, S2, testX



# parallelize_training(S1, S2, test_data, model = model1)
# model_training
# parallelizes 




def train_in_parallel(S1, S2, model_class = None):
    pool = multiprocessing.Pool(processes = 2)

    inputs = [(S1, model_class), (S2, model_class)]
    results = []
    for arg in inputs:
        results.append(pool.apply_async(model_class.train_model, args = arg))
    pool.close()
    pool.join()
    # print(results[0], results[1])
    return results[0].get(), results[1].get()

def train_sequential(S1, S2, model_class = None):
    inputs = [(S1, model_class), (S2, model_class)]
    results = []

    for arg in inputs:
        result = model_class.train_model(*arg)
        results.append(result)

    return results[0], results[1]

# compute_volatility: take l2**2 norm of the difference of predictions
def compute_volatility(model1, model2, test_data):
    test_data = torch.tensor(test_data, dtype = torch.float32)
    return (torch.norm(model1(test_data) - model2(test_data), p = 2)**2).item()

def volatility_experiment(X, y, alpha = 0, test_size = 100, model_class = None):
    S1, S2, testX = generate_train_test_data(X, y, n, alpha, test_size)
    model1, model2 = train_in_parallel(S1, S2, model_class)

    return compute_volatility(model1, model2, test_data=testX)

# detect convergence
def detect_convergence(X, y, alpha: float = 0, rounds = 100, test_size = 25, model_class = None, epsilon = 1e-4):
    total = 0
    running_scores = []
    for i in range(1, rounds):
        val = volatility_experiment(X, y, alpha, test_size, model_class)
        if abs((total/i- (total+val)/(i+1))) < epsilon:
            print(total/i)
            return i, val
        total += val
        running_scores.append(val/i)
    
    return 'Failed to converge in {rounds}', running_scores





# Classical Stability

# E [l(A(S(i),zi))−l(A(S),zi)] where S, z' drawn from data 
# and zi drawn uniformly from S


def stability_one_draw(X, y, n, model_class = None, criterion = None, optimizer = None, lr = 0.01, num_epochs = 100):
    train_data = draw_data(X, y, n)
    z_prime = draw_data(X, y, 1)
    

# draw_data(D, n) --> S

# draw_data(D, 1) -- z' 

# draw_uniformly i

if __name__ == "__main__":
    d = 47
    n = 18857
    Sigma = generate_random_covariance_matrix(d)
    c = np.random.randn(d).reshape(1, d)
    X, y, _ = distribution(Sigma, n, c, variance = 0.2)

    S1, S2, testX = generate_train_test_data(X, y, n, alpha = 0, test_size = 10)
    model1, model2 = train_in_parallel(S1, S2, model_class = LinearRegressor)
    # print(model1.state_dict()['linear.weight'])
    # print(c)
    # _, running_scores = detect_convergence(X, y, alpha = 0, rounds = 10, test_size = 10, model_class = LogisticRegressor, epsilon = 1e-2)
    