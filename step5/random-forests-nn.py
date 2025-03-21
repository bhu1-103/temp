import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Set directories
input_dir = "in/"
output_dir = "out/"
predictions_dir = "predictions_nn/"
os.makedirs(predictions_dir, exist_ok=True)

# Get files
input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
output_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".csv")])

# Define PyTorch model
class NeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(NeuralNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),  # Normalization for stable training
            nn.ReLU(),
            nn.Dropout(0.3),  # Prevent overfitting
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)  # Output layer for regression
        )

    def forward(self, x):
        return self.layers(x)

# Process each file pair
for in_file, out_file in zip(input_files, output_files):
    # Load input (X) using ";" separator
    X = pd.read_csv(os.path.join(input_dir, in_file), delimiter=";")
    X = X.select_dtypes(include=["number"])  # Drop non-numeric columns

    # Load output (y) using "," separator
    y = pd.read_csv(os.path.join(output_dir, out_file), delimiter=",", header=None)
    
    # Transpose y if needed
    if y.shape[0] == 1:
        y = y.T

    # Ensure X and y match
    if X.shape[0] != y.shape[0]:
        print(f"❌ Skipping {in_file}/{out_file}: Shape mismatch! X={X.shape}, y={y.shape}")
        continue

    print(f"✅ Processing {in_file}/{out_file}: X={X.shape}, y={y.shape}")

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1).to(device)

    # Initialize model
    model = NeuralNet(input_dim=X_train.shape[1]).to(device)

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    epochs = 200  # Increased training for better learning
    batch_size = 32
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        y_pred_train = model(X_train_tensor)
        loss = criterion(y_pred_train, y_train_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Evaluate model
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test_tensor).cpu().numpy().flatten()

    # Compute Metrics
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    print(f"📊 {in_file} Metrics:")
    print(f"   - MAE  = {mae:.4f}")
    print(f"   - RMSE = {rmse:.4f}")
    print(f"   - R²   = {r2:.4f}")

    # Predict full dataset
    with torch.no_grad():
        y_full_pred = model(torch.tensor(scaler.transform(X), dtype=torch.float32).to(device)).cpu().numpy().flatten()

    # Save predictions
    pred_file = os.path.join(predictions_dir, f"pred_{out_file}")
    pd.DataFrame(y_full_pred).to_csv(pred_file, index=False, header=False, sep=",")
    print(f"📂 Saved predictions to {pred_file}")

print("🎯 Done!")

