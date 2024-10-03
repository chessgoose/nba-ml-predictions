"""
TODO:
- Remember to remove rows with NaN/missing values 

"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from utils.dataloading import load_regression_data, drop_regression_stats, load_2023_data
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
import matplotlib.pyplot as plt

class QuantileLoss(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        errors = target.unsqueeze(1) - preds
        losses = torch.max((self.quantiles - 1) * errors, self.quantiles * errors)
        return torch.mean(losses)

class QuantileRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc3 = nn.Linear(32, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        #x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=1000, early_stopping_rounds=30):
    best_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        if epoch % 20 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Val Loss: {val_loss}")

        if val_loss < best_loss:
            best_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_rounds:
            print('Early stopping!')
            torch.save(model.state_dict(), f'nn_models/wnba/NN_{str(round(val_loss, 2))}.pth')
            break


def load_data(data, points, test_size=0.2, batch_size=32):
    x_train, x_test, y_train, y_test = train_test_split(data, points, test_size=test_size, shuffle=True)

    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit the scaler on the training data
    x_train = scaler.fit_transform(x_train)
    
    # Transform the test data using the fitted scaler
    x_test = scaler.transform(x_test)
    
    # Save the scaler for future use
    joblib.dump(scaler, 'nn_models/scaler.pkl')

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, y_test

def main():
    league = "wnba"

    random_seed = 69
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    data = load_2023_data()
    points = data["Points"]
    print(data.head())

    drop_regression_stats(data)
    data.drop(["Points"], axis=1, inplace=True)
    print(data.head())

    quantiles = np.array([float(4/9), float(5/9)])

    acc_results = []
    for _ in tqdm(range(15)):
        train_loader, val_loader, y_test = load_data(data, points)

        input_dim = data.shape[1]
        output_dim = len(quantiles)
        model = QuantileRegressor(input_dim, output_dim)
        criterion = QuantileLoss(torch.tensor(quantiles))
        optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

        train_model(model, criterion, optimizer, train_loader, val_loader)

        model.load_state_dict(torch.load('nn_models/wnba/NN.pth'))
        model.eval()

        all_preds = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                outputs = model(inputs)
                all_preds.append(outputs)

        predictions = torch.cat(all_preds).numpy()

        y_lower = predictions[:, 0]  # alpha=0.476
        y_upper = predictions[:, 1]  # alpha=0.524

        predictions = (y_lower + y_upper) / 2
        mae = mean_absolute_error(y_test, predictions)
        print(f"MAE: {mae}")


if __name__ == "__main__":
    main()
