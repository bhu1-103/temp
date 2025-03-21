import os
import pandas as pd

# Define input and output directories
input_dir = "../step2/z_output/"
output_dir = "in/"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all CSV files in input_dir
csv_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

for file in csv_files:
    input_path = os.path.join(input_dir, file)
    output_path = os.path.join(output_dir, file)

    # Read CSV file
    df = pd.read_csv(input_path, delimiter=";")  # Adjust delimiter if needed

    # Calculate variance for all numeric columns
    variances = df.var(numeric_only=True)

    # Print variance for reference
    print(f"\n=== Variance in {file} ===")
    print(variances)

    # Identify zero-variance columns and remove them
    zero_variance_cols = variances[variances == 0].index.tolist()
    
    # Manually remove central frequency
    if "central_freq(GHz)" in df.columns:
        zero_variance_cols.append("central_freq(GHz)")
    
    print(f"Removing columns: {zero_variance_cols}")

    df.drop(columns=zero_variance_cols, inplace=True)

    # Keep node_code but create a new column "is_ap" (1 if AP, 0 if STA)
    df["is_ap"] = df["node_code"].apply(lambda x: 1 if "AP" in x else 0)

    # Save cleaned CSV
    df.to_csv(output_path, index=False, sep=";")

    print(f"Processed {file} -> Saved to {output_path}")

print("✅ Done! Cleaned files are in step5/in/")
