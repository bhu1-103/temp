import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess the data
# Replace these placeholders with your actual data-loading logic
X = torch.rand(1000, 8)  # Example input data (1000 samples, 8 features)
y = torch.rand(1000, 128)  # Example output data (128 target values per sample)

# Normalize the input and output
X = (X - X.mean(dim=0)) / X.std(dim=0)  # Standardize input
y = (y - y.min(dim=0).values) / (y.max(dim=0).values - y.min(dim=0).values)  # Min-max scaling

# Reshape for CNN input (samples, channels, features)
X_cnn = X.unsqueeze(1)  # Add channel dimension for CNN

# Create DataLoader for training
dataset = TensorDataset(X_cnn, X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# --- Define the Hybrid Model ---
class HybridCNN_DNN(nn.Module):
    def __init__(self):
        super(HybridCNN_DNN, self).__init__()
        # CNN Branch
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),  # (Batch, 32, 8)
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1),  # (Batch, 64, 8)
            nn.ReLU(),
            nn.Flatten()  # Flatten to (Batch, 64*8)
        )
        # DNN Branch
        self.dnn = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.ReLU()
        )
        # Final Merge
        self.fc = nn.Linear(64 * 8 + 128, 128)  # Merged output layer

    def forward(self, cnn_input, dnn_input):
        # Forward pass through CNN
        cnn_output = self.cnn(cnn_input)  # Output shape: (Batch, 64*8)
        # Forward pass through DNN
        dnn_output = self.dnn(dnn_input)  # Output shape: (Batch, 128)
        # Concatenate CNN and DNN outputs
        combined = torch.cat((cnn_output, dnn_output), dim=1)
        # Final output layer
        output = self.fc(combined)
        return output

# Instantiate the model
model = HybridCNN_DNN()

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training Loop ---
epochs = 50
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for cnn_input, dnn_input, targets in dataloader:
        # Forward pass
        outputs = model(cnn_input, dnn_input)
        loss = criterion(outputs, targets)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

# --- Evaluation ---
model.eval()
with torch.no_grad():
    # Example evaluation on training data
    cnn_input = X_cnn
    dnn_input = X
    predictions = model(cnn_input, dnn_input)  # Predicted output
    print("Sample Predictions:", predictions[:5])  # Print first 5 predictions
