#!/bin/bash

mkdir -p ~/brain_meningioma/bet_output_except_test

test_list=$(cut -d',' -f1 ~/brain_meningioma/test_files.csv)

for file in ~/brain_meningioma/raw_data/all_t1c/*.nii.gz; do
  filename=$(basename "$file")

  if ! echo "$test_list" | grep -q "$filename"; then
    echo "추론 중: $filename"
    hd-bet \
      -i "$file" \
      -o ~/brain_meningioma/bet_output_except_test/"${filename%.nii.gz}_bet.nii.gz" \
      -device cuda \
      --save_bet_mask
  else
    echo "건너뜀 (test): $filename"
  fi
done
