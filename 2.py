import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np

# --- Custom Dataset ---
class CustomDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.tensor(self.inputs[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y


# --- Load All Input and Output Data ---
# File paths
input_path = "step2.5/reduced_z/"
output_path = "step4/sce1a_output/throughput/"

# Initialize lists to hold combined data
all_features = []
all_targets = []

# Loop through files 000 to 099
for i in range(100):
    input_file = os.path.join(input_path, f"input_nodes_copy_deployment_{i:03}.csv")
    output_file = os.path.join(output_path, f"throughput_{i:03}.csv")
    
    # Check if files exist
    if not os.path.exists(input_file) or not os.path.exists(output_file):
        print(f"Missing file: {input_file} or {output_file}")
        continue

    # Load the input data
    input_df = pd.read_csv(input_file, sep='\s+')  # Fixed deprecated parameter

    # Extract numeric and categorical columns
    numeric_cols = ["x(m)", "y(m)", "z(m)"]  # Specify numeric columns
    categorical_cols = ["node_type", "wlan_code"]  # Specify categorical columns

    # Numeric feature preprocessing
    scaler = StandardScaler()
    numeric_data = scaler.fit_transform(input_df[numeric_cols])

    # Categorical feature preprocessing (One-Hot Encoding)
    encoder = OneHotEncoder(sparse_output=False)  # Fixed parameter name
    categorical_data = encoder.fit_transform(input_df[categorical_cols])

    # Combine numeric and categorical features
    features = np.hstack([numeric_data, categorical_data])

    # Load the output data
    output_df = pd.read_csv(output_file, header=None)
    targets = output_df.values.flatten()  # Flatten to 1D array

    # Ensure features and targets align
    if features.shape[0] != len(targets):
        print(f"Mismatch between input rows and output values for file {i:03}")
        continue

    # Append to the combined data
    all_features.append(features)
    all_targets.append(targets)

# Combine all data into a single array
all_features = np.vstack(all_features)  # Stack vertically
all_targets = np.hstack(all_targets)  # Stack horizontally

# --- Train-Test Split ---
X_train, X_val, y_train, y_val = train_test_split(all_features, all_targets, test_size=0.2, random_state=42)

# Create datasets and dataloaders
train_dataset = CustomDataset(X_train, y_train)
val_dataset = CustomDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# --- Define the Hybrid Model ---
class HybridCNN_DNN(nn.Module):
    def __init__(self, input_dim):
        super(HybridCNN_DNN, self).__init__()
        # CNN Branch
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # DNN Branch
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Final Merge
        self.fc = nn.Linear(64 * all_features.shape[1] + 128, 1)  # Output single value per input row

    def forward(self, dnn_input):
        # Reshape DNN input for CNN branch
        cnn_input = dnn_input.unsqueeze(1)  # Add channel dimension for CNN (Batch, 1, Features)
        # Forward pass through CNN
        cnn_output = self.cnn(cnn_input)  # Output shape: (Batch, 64 * Features)
        # Forward pass through DNN
        dnn_output = self.dnn(dnn_input)  # Output shape: (Batch, 128)
        # Concatenate CNN and DNN outputs
        combined = torch.cat((cnn_output, dnn_output), dim=1)
        # Final output layer
        output = self.fc(combined)
        return output


# --- Training Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = HybridCNN_DNN(input_dim=all_features.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)  # Squeeze to match target shape
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            val_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")


# --- Save the Model ---
#torch.save(model.state_dict(), "hybrid_cnn_dnn_model.pth")

# --- Prediction Example ---
model.eval()
sample_input = torch.tensor(X_val[:50], dtype=torch.float32).to(device)
predictions = model(sample_input).cpu().detach().numpy()
print("Predictions (first 5):", predictions)
