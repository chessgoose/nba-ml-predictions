import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Function to create sequences from time series data
def create_sequences(data, sequence_length):
    sequences = []
    for i in range(len(data) - sequence_length):
        sequence = data[i : (i + sequence_length)]
        sequences.append(sequence)
    return torch.tensor(sequences, dtype=torch.float32)

# Generate sample time series data (replace this with your own data)
np.random.seed(42)
num_series = 5
num_points = 100
data = np.cumsum(np.random.randn(num_points, num_series), axis=0)

# Visualize the time series data
for i in range(num_series):
    plt.plot(data[:, i], label=f"Series {i + 1}")

plt.title("Multiple Time Series Data")
plt.legend()
plt.show()

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_normalized = scaler.fit_transform(data)

# Choose the sequence length
sequence_length = 10

# Create sequences for training
sequences = create_sequences(data_normalized, sequence_length)

# Split the data into training and testing sets
train_size = int(len(sequences) * 0.8)
train, test = sequences[:train_size], sequences[train_size:]

# Split each sequence into input (X) and output (y)
X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Hyperparameters
input_size = num_series
hidden_size = 50
output_size = 1
num_epochs = 50
learning_rate = 0.001

# Initialize the model, loss function, and optimizer
model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_loss = criterion(test_outputs, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# Invert the normalization to get the actual values
predictions_actual = scaler.inverse_transform(test_outputs.numpy())
y_test_actual = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))

# Visualize the predictions
plt.plot(y_test_actual, label='Actual')
plt.plot(predictions_actual, label='Predicted', linestyle='--', color='red')
plt.title("PyTorch LSTM Predictions on Test Data")
plt.legend()
plt.show()
