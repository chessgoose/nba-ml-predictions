# Links
# https://towardsdatascience.com/deep-quantile-regression-c85481548b5a
# https://link.springer.com/article/10.1007/s10489-022-03958-7

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from dataloading import load_regression_data, drop_regression_stats
import warnings
import matplotlib.pyplot as plt

# Set the random seed for reproducibility
random_seed = 1234
np.random.seed(random_seed)
torch.manual_seed(random_seed)

league = "wnba"
warnings.simplefilter(action='ignore', category=FutureWarning)

data = load_regression_data(league)
lines = data['Line']
OU = data['OU Result']
points = data['Points']
print(data.head())

drop_regression_stats(data)
data.drop(["Relative Strength"], axis=1, inplace=True)
data.drop(["OU Result", "Line", "Points"], axis=1, inplace=True)
print(data.head())

quantiles = [0.476, 0.524]

# Define the neural network model
class QuantileRegressionNN(nn.Module):
    def __init__(self, input_dim, layer_dim):
        super(QuantileRegressionNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, layer_dim)
        self.fc2 = nn.Linear(layer_dim, layer_dim)
        #self.dropout = nn.Dropout(0.2)
        self.quantile_476 = nn.Linear(layer_dim, 1)
        self.quantile_524 = nn.Linear(layer_dim, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = self.dropout(x)
        q476 = self.quantile_476(x)
        q524 = self.quantile_524(x)
        return q476, q524

# Define custom quantile loss
def quantile_loss(quantile, y_true, y_pred):
    e = y_true - y_pred
    return torch.mean(torch.max(quantile * e, (quantile - 1) * e))

# Convert data to PyTorch tensors
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


acc_results = []
for x in tqdm(range(15)):
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(data, points, lines, test_size=.2, shuffle=True)

    x_train = torch.tensor(x_train.values, dtype=torch.float32).to(device)
    x_test = torch.tensor(x_test.values, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1).to(device)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1).to(device)
    z_test = torch.tensor(z_test.values, dtype=torch.float32).unsqueeze(1).to(device)

    model = QuantileRegressionNN(input_dim=x_train.shape[1], layer_dim=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 1000
    patience = 500
    best_loss = float('inf')
    epochs_no_improve = 0
    best_model_wts = model.state_dict()

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        q476_pred, q524_pred = model(x_train)
        loss = (quantile_loss(quantiles[0], y_train, q476_pred) + quantile_loss(quantiles[1], y_train, q524_pred)) / 2
        loss.backward()
        optimizer.step()

        # Early stopping
        model.eval()
        with torch.no_grad():
            q476_pred_val, q524_pred_val = model(x_test)
            val_loss = (quantile_loss(quantiles[0], y_test, q476_pred_val) + quantile_loss(quantiles[1], y_test, q524_pred_val)) / 2

        if val_loss.item() < best_loss:
            best_loss = val_loss.item()
            best_model_wts = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    model.eval()
    with torch.no_grad():
        predictions = model(x_test)
    y_lower = predictions[0].cpu().numpy().flatten()  # alpha=0.476
    y_upper = predictions[1].cpu().numpy().flatten()  # alpha=0.524

    valid_indices = np.where((z_test.cpu().numpy().flatten() < y_lower) | (z_test.cpu().numpy().flatten() > y_upper))[0]

    if len(valid_indices) == 0:
        print("No valid predictions outside the range [predictions_low, predictions_high]")
        continue

    valid_predictions = (y_lower[valid_indices] + y_upper[valid_indices]) / 2
    valid_y_test = y_test.cpu().numpy().flatten()[valid_indices]
    valid_z_test = z_test.cpu().numpy().flatten()[valid_indices]

    mae = mean_absolute_error(valid_y_test, valid_predictions)
    print(f"MAE: {mae}")

    # Compare the predictions to the lines to compute accuracy when compared to over/under
    predicted_ou_results = np.where(valid_predictions > valid_z_test, 1, 0)
    actual_ou_results = np.where(valid_y_test > valid_z_test, 1, 0)
    acc = round(np.mean(predicted_ou_results == actual_ou_results) * 100, 1)
    print(f"Accuracy: {acc}% on {len(predicted_ou_results)} results")
    acc_results.append(acc)

    # Only save results if they are the best so far
    if acc == max(acc_results):
        torch.save(model.state_dict(), f'models/regression/{league}/DNN_{acc}%_OU.pth')

        plt.figure(figsize=(10, 6))
        plt.scatter(valid_y_test, valid_predictions, color='blue', label='Predictions')

        # Plot a line of perfect prediction for reference
        min_val = min(min(valid_y_test), min(valid_predictions))
        max_val = max(max(valid_y_test), max(valid_predictions))
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='Perfect Prediction')

        # Add labels and title
        plt.xlabel('Actual Values (y_test)')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.legend()

        # Show the plot
        plt.show()