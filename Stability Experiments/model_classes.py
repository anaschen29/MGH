import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import RandomForestClassifier
import torch
from tqdm import tqdm

class LinearRegressor(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

    
    def train_model(S, model_class = None, criterion = None, optimizer = None, lr= 0.01, num_epochs = 100):
        X, y = S
        # make this optional
        X = torch.tensor(X, dtype = torch.float32)
        y = torch.tensor(y, dtype = torch.float32)
            
        model = LinearRegressor(X.shape[1])
        
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
    

    def train_model(S, model_class = None, criterion = None, optimizer = None, lr= 0.01, num_epochs = 100):
        X, y = S
        # make this optional
        X = torch.tensor(X, dtype = torch.float32)
        y = torch.tensor(y, dtype = torch.float32)
        n_epochs = 100

        criterion = nn.BCELoss()
        model = LogisticRegressor(X.shape[1])

        for epoch in tqdm(range(n_epochs)):
            model.train()
            
            # Forward pass
            outputs = model(X)
            loss = criterion(outputs, y)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}')
        
        return model

    
class RandomForest:
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=random_state
        )

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)