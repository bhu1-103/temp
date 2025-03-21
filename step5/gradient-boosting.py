import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV

# Set directories
input_dir = "in/"
output_dir = "out/"
predictions_dir = "predictions_xgb/"
os.makedirs(predictions_dir, exist_ok=True)

# Get files
input_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".csv")])
output_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".csv")])

for in_file, out_file in zip(input_files, output_files):
    # Load input (X)
    X = pd.read_csv(os.path.join(input_dir, in_file), delimiter=";")
    X = X.select_dtypes(include=["number"])  # Keep only numeric

    # Load output (y)
    y = pd.read_csv(os.path.join(output_dir, out_file), delimiter=",", header=None)

    # Fix y shape
    if y.shape[0] == 1:
        y = y.T

    if X.shape[0] != y.shape[0]:
        print(f"❌ Skipping {in_file}/{out_file}: Shape mismatch! X={X.shape}, y={y.shape}")
        continue

    print(f"✅ Processing {in_file}/{out_file}: X={X.shape}, y={y.shape}")

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter tuning
    param_dist = {
        "n_estimators": randint(100, 500),
        "max_depth": randint(3, 10),
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0]
    }

    model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
    search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=10, cv=3, scoring="neg_root_mean_squared_error", verbose=1, n_jobs=-1)
    
    search.fit(X_train_scaled, y_train.values.ravel())

    # Best model
    best_model = search.best_estimator_

    # Predict & Evaluate
    y_pred = best_model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"📊 {in_file} Metrics (Optimized XGBoost):")
    print(f"   - MAE  = {mae:.4f}")
    print(f"   - RMSE = {rmse:.4f}")
    print(f"   - R²   = {r2:.4f}")

    # Predict full dataset
    y_full_pred = best_model.predict(scaler.transform(X))

    # Save predictions
    pred_file = os.path.join(predictions_dir, f"pred_{out_file}")
    pd.DataFrame(y_full_pred).to_csv(pred_file, index=False, header=False, sep=",")
    print(f"📂 Saved predictions to {pred_file}")

print("🎯 Done!")

