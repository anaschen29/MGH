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
# helper for both

## draw_data(D, n, with_replacement = True)

def draw_data(X, y, n, with_replacement = True):
    num_samples = X.shape[0]
    indices = np.random.choice(num_samples, size = n, replace = True)
    return X[indices], y[indices]

def generate_train_test_data(X, y, n, alpha = 0, test_size = 100):
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


def train_model(S, model_class = None, criterion = None, optimizer = None, lr= 0.01, num_epochs = 100):
    X, y = S
    model = model_class()

    # make this optional
    X = torch.tensor(X, dtype = torch.float32)
    y = torch.tensor(y, dtype = torch.float32)

    if model_class == None:
        model_class = LinearRegressor

    if criterion == None:
        criterion = nn.MSELoss()

    if optimizer == None:
        optimizer = optim.SGD(model.parameters(), lr = lr)

    for _ in tqdm(range(num_epochs)):
        y_pred = model(X)

        loss = criterion(y_pred, y)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    return model

def train_in_parallel(S1, S2, model_class = None):
    pool = multiprocessing.Pool(processes = 2)

    inputs = [(S1, model_class), (S2, model_class)]
    results = []
    for arg in inputs:
        results.append(pool.apply_async(train_model, args = arg))
    pool.close()
    pool.join()

    return results[0].get(), results[1].get()

# compute_volatility: take l2**2 norm of the difference of predictions
def compute_volatility(model1, model2, test_data):
    test_data = torch.tensor(test_data, dtype = torch.float32)
    print(type(model1(test_data)))
    return np.linalg.norm(model1(test_data) - model2(test_data))**2

def experiment(X, y, alpha = 0, test_size = 100, model_class = None):
    S1, S2, testX = generate_train_test_data(X, y, n, alpha, test_size)

    model1, model2 = train_in_parallel(S1, S2, model_class)

    return compute_volatility(model1, model2, test_data=testX )

# detect convergence
def detect_convergence(X, y, alpha = 0, rounds = 10000, test_size = 100, model_class = None, epsilon = 1e-4):
    total = 0
    for i in range(rounds):
        val = experiment(X, y, alpha, test_size, model_class)
        if abs((total/i- (total+i)/(i+2))) < epsilon:
            return i, val
        total += val
    return 'Failed to converge in {rounds}', total/i





# Classical Stability

# E [l(A(S(i),zi))−l(A(S),zi)] where S, z' drawn from data 
# and zi drawn uniformly from S

# draw_data(D, n) --> S

# draw_data(D, 1) -- z' 

# draw_uniformly i

if __name__ == "__main__":
    d = 47
    n = 18857
    Sigma = generate_random_covariance_matrix(d)
    c = np.random.randn(d).reshape(1, d)
    X, y, _ = distribution(Sigma, n, c, variance = 0.2)
    experiment(X, y, alpha = 0, test_size = 100, model_class = LinearRegressor)