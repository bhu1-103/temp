import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set directories
input_dir = "in/"
output_dir = "out/"
predictions_dir = "predictions_xgb/"
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

    # Normalize input features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Convert to XGBoost DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # XGBoost Parameters (Optimized)
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "learning_rate": 0.05,
        "max_depth": 6,
        "n_estimators": 300,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "tree_method": "hist"
    }

    # Train XGBoost model
    model = xgb.train(params, dtrain, num_boost_round=300, evals=[(dtest, "Test")], verbose_eval=10)

    # Predict on test set
    y_pred = model.predict(dtest)

    # Evaluate model
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"📊 {in_file} XGBoost Metrics:")
    print(f"   - MAE  = {mae:.4f}")
    print(f"   - RMSE = {rmse:.4f}")
    print(f"   - R²   = {r2:.4f}")

    # Predict full dataset
    dfull = xgb.DMatrix(X_scaled)
    y_full_pred = model.predict(dfull)

    # Save predictions using "," separator
    pred_file = os.path.join(predictions_dir, f"pred_{out_file}")
    pd.DataFrame(y_full_pred).to_csv(pred_file, index=False, header=False, sep=",")
    print(f"📂 Saved predictions to {pred_file}")

print("🎯 Done!")
