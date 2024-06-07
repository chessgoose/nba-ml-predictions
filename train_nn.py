import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from sklearn.model_selection import train_test_split
from dataloading import load_data
from torch.utils.data import DataLoader, TensorDataset

league = "nba"
random_seed = 1234
torch.manual_seed(random_seed)

# Define the neural network model (assuming NeuralNet is already defined)
class NeuralNet(nn.Module):
    def __init__(self, input_size):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.2)  # 50% dropout
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(self.bn1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out

# Load data
X, y = load_data(league)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, shuffle=True)

# Scale the features
"""
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
"""

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create TensorDataset and DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=25, shuffle=True)

# Initialize the model, loss function, and optimizer
num_features = X_train.shape[1]
model = NeuralNet(num_features)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=5e-4)

# Train the model
# Consider minibatched vs not batched training
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    # implement early stopping
    
    if (epoch + 1) % 5 == 0:
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}')

# Save the model for future use
torch.save(model.state_dict(), f'nn_models/{league}/model.pth')

# Test on evaluation set 
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

model.eval()
with torch.no_grad():
    test_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        
        # Calculate accuracy
        predicted = (outputs >= 0.5).float()
        correct_predictions += (predicted == labels).sum().item()
        total_predictions += labels.size(0)

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = correct_predictions / total_predictions
    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')


"""
import sqlite3
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

current_time = str(time.time())

tensorboard = TensorBoard(log_dir='../../Logs/{}'.format(current_time))
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('../../Models/Trained-Model-OU-' + current_time, save_best_only=True, monitor='val_loss', mode='min')

dataset = "dataset_2012-24"
con = sqlite3.connect("../../Data/dataset.sqlite")
data = pd.read_sql_query(f"select * from \"{dataset}\"", con, index_col="index")
con.close()

OU = data['OU-Cover']
total = data['OU']
data.drop(['Score', 'Home-Team-Win', 'TEAM_NAME', 'Date', 'TEAM_NAME.1', 'Date.1', 'OU-Cover', 'OU'], axis=1, inplace=True)

data['OU'] = np.asarray(total)
data = data.values
data = data.astype(float)

x_train = tf.keras.utils.normalize(data, axis=1)
y_train = np.asarray(OU)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu6))
# model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu6))
model.add(tf.keras.layers.Dense(3, activation=tf.nn.softmax))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=50, validation_split=0.1, batch_size=32, callbacks=[tensorboard, earlyStopping, mcp_save])

print('Done')

"""