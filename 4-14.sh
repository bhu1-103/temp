#!/bin/bash

base_dir="100-files"

for dir in "$base_dir"/1*; do
  if [[ -d "$dir" ]]; then
    a_value="${dir##*-}"
    
    if [[ "$a_value" == "04" || "$a_value" == "14" ]]; then
      echo "Generating data at $dir"
      
      (
        cd "$dir/step1" && ./step1.sh
        cd ../step2 && ./step2.sh
        cd ../step3 && ./combine.sh
        cd ../step4 && ./saigo-no-steppu.sh
      )
      
      echo "Generated data at $dir"
    else
      echo "Skipping $dir (a = $a_value)"
    fi
  fi
done

echo "Completed dataset generation... wake up!!!"
