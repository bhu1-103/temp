#!/bin/bash

source_dir="backup"
dest_base="100-files"

x_values=(20 40 60 80 100)
y_values=(20 40 60 80 100)

mkdir -p 100-files
for x in "${x_values[@]}"; do
  for y in "${y_values[@]}"; do
    for a in {4..14}; do
      formatted_x=$(printf "%03d" $x)
      formatted_y=$(printf "%03d" $y)
      formatted_a=$(printf "%02d" $a)
      
      dest_dir="$dest_base/$formatted_x-$formatted_y-$formatted_a"
      
      echo "Copying $source_dir to $dest_dir..."
      cp -r "$source_dir" "$dest_dir"
      
      csv_file="$dest_dir/step1/input-java.csv"
      
      if [[ -f "$csv_file" ]]; then
        echo "Modifying $csv_file: Setting row 2, columns 1, 2, and 3..."
        
        awk -F';' -v OFS=';' -v x="$formatted_x" -v y="$formatted_y" -v a="$formatted_a" \
        'NR == 2 {$1 = x; $2 = y; $3 = a} {print}' "$csv_file" > tmp && mv tmp "$csv_file"

        echo "Modified $csv_file successfully!"
      else
        echo "Warning: $csv_file not found!"
      fi
    done
  done
done

echo "Copy and modification completed!"
