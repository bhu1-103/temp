import torch
import numpy as np
import pandas as pd

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the same TinyCNN model structure
class TinyCNN(torch.nn.Module):
    def __init__(self):
        super(TinyCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(32 * 10 * 10, 128)
        self.fc2 = torch.nn.Linear(128, 100)  # Output size matches your target size

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, kernel_size=2)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load the trained model
model = TinyCNN()
model.load_state_dict(torch.load("model.pth"))  # Replace with your checkpoint filename
model.to(device)
model.eval()  # Set model to evaluation mode

# Load the input data
input_csv = "step2.5/reduced_z/input_nodes_copy_deployment_000.csv"  # Replace with your input file path
input_df = pd.read_csv(input_csv, delim_whitespace=True)  # Assuming whitespace delimiter
input_data = input_df.iloc[:, 3:].values  # Select only the numerical features (x, y, z, etc.)

# Preprocess the input data
# Convert input to the shape (batch_size, channels, height, width)
# For example, reshape to (1, 1, 20, 20) if you have 20x20 input
input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

# Make predictions
with torch.no_grad():
    predictions = model(input_tensor)

# Convert predictions to numpy and save
predicted_output = predictions.cpu().numpy()

# Save to a CSV file
output_csv = "predictions.csv"
np.savetxt(output_csv, predicted_output, delimiter=",")
print(f"Predictions saved to {output_csv}")
