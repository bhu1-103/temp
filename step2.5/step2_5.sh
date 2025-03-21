#!/usr/bin/zsh

input_dir="../step2/z_output"
output_dir="../step2.5/reduced_z"
mkdir -p "$output_dir"

for file in "$input_dir"/*.csv; do
    base_name=$(basename "$file")
    awk -F ";" '{print $1,$2,$3,$4,$5,$6,$10,$11}' "$file" > "$output_dir/$base_name"
    echo "Processed $base_name -> $output_dir/$base_name"
done
