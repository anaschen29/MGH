import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
import torch
from tqdm import tqdm

class LinearRegressor(nn.Module):
    def __init__(self, input_dim, output_dim = 1):
        super(LinearRegressor, self).__init__()
        self.model = nn.Linear(input_dim, output_dim)


    def forward(self, x):
        return self.model(x)

    
    def train_model(self, S, lr= 0.0001, num_epochs = 100):
        X, y = S
        # make this optional
        X = torch.tensor(X, dtype = torch.float32)
        y = torch.tensor(y, dtype = torch.float32)
    
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.model.parameters(), lr = lr)

        for _ in tqdm(range(num_epochs)):
            y_pred = self.model(X)

            loss = criterion(y_pred, y)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        return self.model

    def evaluate(self, X_test, y_test):
        # Evaluation mode
        self.eval()
        with torch.no_grad():
            test_outputs = self.forward(X_test)
            criterion = nn.MSELoss()
            test_loss = criterion(test_outputs, y_test)
            print(f'Test Loss: {test_loss.item():.4f}')
            return test_loss.item()
    
class LogisticRegressor(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LogisticRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.linear(x))
    

    def train_model(self, S, lr= 0.01, num_epochs = 100):
        X, y = S
        # make this optional
        X = torch.tensor(X, dtype = torch.float32)
        y = torch.tensor(y, dtype = torch.float32)
        y = y.unsqueeze(1)
        y = y.view(-1)
        n_epochs = 100

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        # Ensure that X and y are tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.float32)

        for epoch in range(num_epochs):
            # Forward pass
            outputs = self(X).squeeze()
            loss = criterion(outputs, y)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size = 64, output_size = 1):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.input_size = input_size
        self.model = lambda x: self.forward(x)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    def train_model(self, S, lr = 0.0001, num_epochs = 1000):
        X, y = S
        # make this optional
        X = torch.tensor(X, dtype = torch.float32)
        y = torch.tensor(y, dtype = torch.float32)
        y = y.unsqueeze(1)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr)

        for epoch in range(num_epochs):
            # Forward pass
            outputs = self(X)
            loss = criterion(outputs, y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 100 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# from Aladdin Persson's youtube
class LeNet(nn.Module):
  def __init__(self, activation = nn.ReLU()):
    super(LeNet, self).__init__()
    
    self.pool = nn.AvgPool2d(kernel_size = (2,2), stride = (2,2))
    self.conv1 = nn.Conv2d(in_channels = 1, out_channels =6, kernel_size = (5,5), stride = (1,1), padding = (0,0))
    self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, kernel_size = (5,5), stride = (1,1), padding = (0,0))
    self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, kernel_size = (5,5), stride = (1,1), padding = (0,0))
    self.linear1 = nn.Linear(120, 84)
    self.linear2 = nn.Linear(84, 10)

    self.activation = activation
    # self.model = lambda x: self.forward(x)

  def forward(self, x):
    out = self.conv1(x)
    out = self.activation(out)
    out = self.pool(out)
    out = self.conv2(out)
    out = self.activation(out)
    out = self.pool(out)
    out = self.conv3(out)
    out = self.activation(out)
    out = torch.flatten(out, 1)
    out = self.linear1(out)
    out = self.activation(out)
    out = self.linear2(out)
    return out

  
  def train_model(self, train_loader, lr = 0.01, num_epochs = 50):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(self.parameters(), lr)
    self.train()
    for _ in tqdm(range(num_epochs)):
      self.train()
      for _, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = self(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    

    
