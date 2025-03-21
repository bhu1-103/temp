import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set directories
input_dir = "in/"
output_dir = "out/"
predictions_dir = "predictions_nn/"
os.makedirs(predictions_dir, exist_ok=True)

# Get files
input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
output_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".csv")])

for in_file, out_file in zip(input_files, output_files):
    # Load input (X) using ";" separator
    X = pd.read_csv(os.path.join(input_dir, in_file), delimiter=";")

    # Drop non-numeric columns
    X = X.select_dtypes(include=["number"])

    # Load output (y) using "," separator
    y = pd.read_csv(os.path.join(output_dir, out_file), delimiter=",", header=None)

    # Fix y shape: Transpose if needed (1, N) → (N, 1)
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

    # Build the Neural Network Model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2),  # Regularization to prevent overfitting
        Dense(32, activation='relu'),
        Dense(1)  # Output layer (Regression)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train the model
    model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test),
              epochs=100, batch_size=16, verbose=1)

    # Evaluate on test data
    y_pred = model.predict(X_test_scaled).flatten()

    # Compute Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"📊 {in_file} Metrics:")
    print(f"   - MAE  = {mae:.4f}")
    print(f"   - RMSE = {rmse:.4f}")
    print(f"   - R²   = {r2:.4f}")

    # Predict full dataset
    y_full_pred = model.predict(scaler.transform(X)).flatten()

    # Save predictions
    pred_file = os.path.join(predictions_dir, f"pred_{out_file}")
    pd.DataFrame(y_full_pred).to_csv(pred_file, index=False, header=False, sep=",")
    print(f"📂 Saved predictions to {pred_file}")

print("🎯 Done!")

