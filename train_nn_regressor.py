"""
TODO:
- Remember to remove rows with NaN/missing values since that will fuck things up

"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataloading import load_regression_data, drop_regression_stats
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
        x = self.fc3(x)
        return x

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=1000, early_stopping_rounds=40):
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
            torch.save(model.state_dict(), 'models/regression/wnba/NN.pth')
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_rounds:
            print('Early stopping!')
            break

def load_data(data, points, lines, test_size=0.2, batch_size=32):
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(data, points, lines, test_size=test_size, shuffle=True)

    # Initialize the scaler
    scaler = StandardScaler()
    
    # Fit the scaler on the training data
    x_train = scaler.fit_transform(x_train)
    
    # Transform the test data using the fitted scaler
    x_test = scaler.transform(x_test)
    
    # Save the scaler for future use
    joblib.dump(scaler, 'scaler.pkl')

    train_dataset = torch.utils.data.TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test.values, dtype=torch.float32))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, y_test, z_test

def main():
    league = "wnba"

    random_seed = 69
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)

    data = load_regression_data(league)
    lines = data['Line']
    OU = data['OU Result']
    points = data['Points']
    print(data.head())

    drop_regression_stats(data)
    data.drop(["OU Result", "Line", "Points"], axis=1, inplace=True)
    print(data.head())

    quantiles = np.array([0.476, 0.524])

    acc_results = []
    for _ in tqdm(range(20)):
        train_loader, val_loader, y_test, z_test = load_data(data, points, lines)

        input_dim = data.shape[1]
        output_dim = len(quantiles)
        model = QuantileRegressor(input_dim, output_dim)
        criterion = QuantileLoss(torch.tensor(quantiles))
        optimizer = optim.Adam(model.parameters(), lr=0.003)

        train_model(model, criterion, optimizer, train_loader, val_loader)

        model.load_state_dict(torch.load('models/regression/wnba/NN.pth'))
        model.eval()

        all_preds = []
        with torch.no_grad():
            for inputs, _ in val_loader:
                outputs = model(inputs)
                all_preds.append(outputs)

        predictions = torch.cat(all_preds).numpy()

        y_lower = predictions[:, 0]  # alpha=0.476
        y_upper = predictions[:, 1]  # alpha=0.524

        padding = 0.5
        valid_indices = np.where((z_test < np.minimum(y_upper, y_lower) - padding) | (z_test > np.maximum(y_lower, y_upper) + padding))[0]

        if len(valid_indices) == 0:
            print("No valid predictions outside the range [predictions_low, predictions_high]")
            continue

        valid_predictions = (y_lower[valid_indices] + y_upper[valid_indices]) / 2
        valid_y_test = y_test.iloc[valid_indices]
        valid_z_test = z_test.iloc[valid_indices]

        mae = mean_absolute_error(valid_y_test, valid_predictions)
        print(f"MAE: {mae}")

        predicted_ou_results = np.where(valid_predictions > valid_z_test, 1, 0)
        actual_ou_results = np.where(valid_y_test > valid_z_test, 1, 0)
        acc = round(np.mean(predicted_ou_results == actual_ou_results) * 100, 1)
        print(f"Accuracy: {acc}% on {len(predicted_ou_results)} results")
        acc_results.append(acc)

        if acc == max(acc_results):
            plt.figure(figsize=(10, 6))
            plt.scatter(valid_y_test, valid_predictions, color='blue', label='Predictions')

            min_val = min(min(valid_y_test), min(valid_predictions))
            max_val = max(max(valid_y_test), max(valid_predictions))
            plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

            plt.xlabel('Actual Values (y_test)')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values')
            plt.legend()
            plt.show()

if __name__ == "__main__":
    main()
