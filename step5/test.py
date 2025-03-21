import os
import pandas as pd

input_dir = "in/"
output_dir = "out/"

for file in sorted(os.listdir(input_dir)):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(input_dir, file), delimiter=";")
        print(f"📂 {file} → Shape: {df.shape}")

for file in sorted(os.listdir(output_dir)):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(output_dir, file), delimiter=",", header=None)
        print(f"📂 {file} → Shape: {df.shape}")

