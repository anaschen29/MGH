import torch.nn as nn
import torch.optim as optim


class LinearRegressor(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)