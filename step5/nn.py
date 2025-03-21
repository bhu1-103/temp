import os
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set directories
input_dir = "in/"
output_dir = "out/"
predictions_dir = "predictions_gnn/"
os.makedirs(predictions_dir, exist_ok=True)

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define GCN Model
class GNNRegressor(torch.nn.Module):
    def __init__(self, input_dim):
        super(GNNRegressor, self).__init__()
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x

for in_file, out_file in zip(sorted(os.listdir(input_dir)), sorted(os.listdir(output_dir))):
    if not in_file.endswith(".csv") or not out_file.endswith(".csv"):
        continue

    # Load data
    X = pd.read_csv(os.path.join(input_dir, in_file), delimiter=";").select_dtypes(include=["number"])
    y = pd.read_csv(os.path.join(output_dir, out_file), delimiter=",", header=None)
    if y.shape[0] == 1: y = y.T  # Fix shape

    if X.shape[0] != y.shape[0]:  # Ensure sizes match
        print(f"❌ Skipping {in_file}: Shape mismatch!")
        continue

    print(f"✅ Processing {in_file}")

    # Scale data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Create graph structure (fully connected)
    num_nodes = X.shape[0]
    edge_index = torch.tensor([(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j], dtype=torch.long).t().contiguous()

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)

    # Train/Test split
    train_idx, test_idx = train_test_split(range(num_nodes), test_size=0.2, random_state=42)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True

    # Convert to PyG Data object
    data = Data(x=X_tensor, edge_index=edge_index, y=y_tensor.squeeze())
    data.train_mask = train_mask
    data.test_mask = test_mask
    data = data.to(device)

    # Initialize model
    model = GNNRegressor(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.MSELoss()

    # Train
    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        y_pred = model(data.x, data.edge_index).cpu().numpy().flatten()
    
    y_test = data.y.cpu().numpy().flatten()[data.test_mask]
    y_pred_test = y_pred[data.test_mask]

    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = mean_squared_error(y_test, y_pred_test, squared=False)
    r2 = r2_score(y_test, y_pred_test)

    print(f"📊 {in_file} GNN Metrics:")
    print(f"   - MAE  = {mae:.4f}")
    print(f"   - RMSE = {rmse:.4f}")
    print(f"   - R²   = {r2:.4f}")

    # Save predictions
    pred_file = os.path.join(predictions_dir, f"pred_{out_file}")
    pd.DataFrame(y_pred).to_csv(pred_file, index=False, header=False, sep=",")
    print(f"📂 Saved predictions to {pred_file}")

print("🎯 Done!")

