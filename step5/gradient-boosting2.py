import os
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Directories
input_dir = "in/"
output_dir = "out/"
predictions_dir = "predictions/"
os.makedirs(predictions_dir, exist_ok=True)

# Get files
input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
output_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".csv")])

for in_file, out_file in zip(input_files, output_files):
    # Load input (X) using ";" separator
    X = pd.read_csv(os.path.join(input_dir, in_file), delimiter=";")
    
    # Drop unnecessary columns
    drop_columns = ['is_ap', 'max_channel_allowed']  # Adjust if needed
    X = X.drop(columns=[col for col in drop_columns if col in X.columns], errors='ignore')
    
    # Select only numeric columns
    X = X.select_dtypes(include=["number"])

    # Load output (y) using "," separator
    y = pd.read_csv(os.path.join(output_dir, out_file), delimiter=",", header=None)

    # Fix y shape: Transpose if needed
    if y.shape[0] == 1:
        y = y.T

    # Ensure X and y match
    if X.shape[0] != y.shape[0]:
        print(f"❌ Skipping {in_file}/{out_file}: Shape mismatch! X={X.shape}, y={y.shape}")
        continue

    print(f"✅ Processing {in_file}/{out_file}: X={X.shape}, y={y.shape}")

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Gradient Boosting Model
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train.values.ravel())  # Flatten y_train

    # Predict on test set
    y_pred = model.predict(X_test)

    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Calculate MAPE & Accuracy
    mape = np.mean(np.abs((y_test.values.ravel() - y_pred) / y_test.values.ravel())) * 100
    accuracy = 100 - mape

    print(f"📊 {in_file} Metrics:")
    print(f"   - MAE  = {mae:.4f}")
    print(f"   - RMSE = {rmse:.4f}")
    print(f"   - R²   = {r2:.4f}")
    print(f"   - MAPE  = {mape:.2f}%")
    print(f"   - Accuracy = {accuracy:.2f}%")

    # Predict full dataset
    y_full_pred = model.predict(X)

    # Save predictions using "," separator
    pred_file = os.path.join(predictions_dir, f"pred_{out_file}")
    pd.DataFrame(y_full_pred).to_csv(pred_file, index=False, header=False, sep=",")
    print(f"📂 Saved predictions to {pred_file}")

print("🎯 Done!")

