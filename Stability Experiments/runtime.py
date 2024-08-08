import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import tqdm
from tqdm import tqdm
from collections import defaultdict
from scipy.optimize import curve_fit
from helpers import *
import multiprocessing
import os
import torch
from model_classes import LinearRegressor, LogisticRegressor, FeedForwardNN, LeNet
import torch.nn as nn
import torch.optim as optim
import warnings
from sklearn.datasets import make_classification
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision import transforms
import torch.optim as optim
from torchvision.transforms import ToTensor

warnings.filterwarnings("ignore", module="torch")
warnings.filterwarnings("ignore", module="torch.utils")

# warnings.filterwarnings("ignore", category = UserWarning)
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


def train_in_parallel(S1, S2, m1, m2, lr = 0.0001):
    mp.set_start_method('spawn')
    processes = []
    p1 = mp.Process(target = type(m1).train_model, args = (m1, S1, lr))
    p2 = mp.Process(target = type(m1).train_model, args = (m2, S2, lr))
    processes.append(p1)
    processes.append(p2)

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    return m1, m2

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
    transform = transforms.Compose([
    transforms.Pad(2),  # Pad with 2 pixels on each side
    transforms.ToTensor(),  # Convert PIL image to tensor
    transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST
                ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    batch_size = 128
    subset_size = 5000

    np.random.seed(12)
    S1_indices = np.random.choice(len(train_dataset), size=subset_size, replace=False)
    S1_dataset = Subset(train_dataset, S1_indices)
    
    np.random.seed(42)
    S2_indices = np.random.choice(len(train_dataset), size=subset_size, replace=False)
    S2_dataset = Subset(train_dataset, S2_indices)

    S1 = DataLoader(S1_dataset, batch_size = 512, shuffle = True)
    S2 = DataLoader(S2_dataset, batch_size = 512, shuffle = True)
    

    m1 = LeNet()
    m2 = LeNet()
    m1, m2 = train_in_parallel(S1, S2, m1, m2)
    test_loader = torch.utils.data.DataLoader(test_dataset)

    with torch.no_grad():
        running = 0
        counter = 0
        for data, _ in test_loader:
            running += np.linalg.norm((m1(data)-m2(data)).detach().numpy())**2
        print(running/len(test_dataset))
    