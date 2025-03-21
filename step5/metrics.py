import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Paths
out_dir = "out/"
pred_dir = "predictions/"

# Load files
out_files = sorted(f for f in os.listdir(out_dir) if f.endswith(".csv"))
pred_files = sorted(f for f in os.listdir(pred_dir) if f.endswith(".csv"))

# Normalize prediction filenames (remove "pred_" prefix)
pred_file_map = {f.replace("pred_", ""): f for f in pred_files}

# Store results
results = []

# Iterate over `out/` files and find matching `predictions/` file
for out_file in out_files:
    if out_file not in pred_file_map:
        print(f"❌ Skipping {out_file}: No matching prediction file found")
        continue

    pred_file = pred_file_map[out_file]

    # Load data
    y_true = pd.read_csv(os.path.join(out_dir, out_file))
    y_pred = pd.read_csv(os.path.join(pred_dir, pred_file))

    # Ensure they have the same shape
    if y_true.shape != y_pred.shape:
        print(f"❌ Skipping {out_file}: Shape mismatch! y_true={y_true.shape}, y_pred={y_pred.shape}")
        continue

    # Convert to NumPy arrays
    y_true = y_true.values.flatten()
    y_pred = y_pred.values.flatten()

    # Compute metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # Handle zero values in y_true for MAPE
    nonzero_mask = y_true != 0
    if np.any(nonzero_mask):
        mape = np.mean(np.abs((y_true[nonzero_mask] - y_pred[nonzero_mask]) / y_true[nonzero_mask])) * 100
        accuracy = 100 - mape
    else:
        mape = np.nan
        accuracy = np.nan

    # Store results
    results.append([out_file, mae, rmse, r2, mape, accuracy])

    # Print results
    print(f"📊 {out_file} Metrics:")
    print(f"   - MAE  = {mae:.4f}")
    print(f"   - RMSE = {rmse:.4f}")
    print(f"   - R²   = {r2:.4f}")
    print(f"   - MAPE = {mape:.2f}%")
    print(f"   - Accuracy = {accuracy:.2f}%\n")

# Save results to CSV
results_df = pd.DataFrame(results, columns=["File", "MAE", "RMSE", "R²", "MAPE", "Accuracy"])
results_df.to_csv("prediction_accuracy_summary.csv", index=False)

print("📂 Accuracy summary saved to prediction_accuracy_summary.csv")
print("🎯 Comparison Done!")
