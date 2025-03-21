import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
INPUT_PATH = "step2.5/reduced_z"
OUTPUT_PATH = "step4/sce1a_output/throughput"
NUM_FILES = 100
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 0.001

# Data Loading
def load_data(input_path, output_path, num_files):
    inputs, outputs = [], []

    for i in range(num_files):
        input_file = os.path.join(input_path, f"input_nodes_copy_deployment_{i:03d}.csv")
        output_file = os.path.join(output_path, f"throughput_{i:03d}.csv")
        try:
            input_df = pd.read_csv(input_file, delim_whitespace=True)
            input_data = input_df.iloc[:, 3:].values  # Numeric columns only
        except Exception as e:
            print(f"Error reading {input_file}: {e}")
            continue

        try:
            output_data = pd.read_csv(output_file, header=None).values.flatten()
        except Exception as e:
            print(f"Error reading {output_file}: {e}")
            continue

        if len(output_data) == input_data.shape[0]:
            inputs.append(input_data)
            outputs.append(output_data)

    inputs = np.vstack(inputs)
    outputs = np.hstack(outputs)

    scaler = StandardScaler()
    inputs = scaler.fit_transform(inputs)

    return inputs, outputs, scaler

# Load data
inputs, outputs, scaler = load_data(INPUT_PATH, OUTPUT_PATH, NUM_FILES)
inputs = inputs.reshape(inputs.shape[0], 1, -1, inputs.shape[1])
X_train, X_test, y_train, y_test = train_test_split(inputs, outputs, test_size=0.2, random_state=42)

# PyTorch DataLoaders
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, pin_memory=True)

# Model Definition

class TinyCNN(torch.nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(32 * 5 * 5, 128)  # Adjust dimensions
        self.fc2 = torch.nn.Linear(128, 1)  # Single output for regression

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = TinyCNN().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_inputs, batch_outputs in train_loader:
            batch_inputs, batch_outputs = batch_inputs.to(device), batch_outputs.to(device)
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions.squeeze(), batch_outputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")

# Evaluation Loop
def evaluate_model(model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_inputs, batch_outputs in test_loader:
            batch_inputs, batch_outputs = batch_inputs.to(device), batch_outputs.to(device)
            predictions = model(batch_inputs)
            loss = criterion(predictions.squeeze(), batch_outputs)
            test_loss += loss.item()
    print(f"Test Loss: {test_loss / len(test_loader):.4f}")

# Train and Evaluate
train_model(model, train_loader, criterion, optimizer, EPOCHS)
evaluate_model(model, test_loader)
torch.save(model.state_dict(), "tiny_cnn_model.pth")
