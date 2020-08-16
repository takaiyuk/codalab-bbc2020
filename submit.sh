#!/bin/bash

# Generate test_prediction.csv for submission from submissions/submission_${run_name}.csv e.g. submission_run000.csv
run_name=$1
echo ${run_name}
submission_filepath=submissions/submission_${run_name}.csv

if [ -f "${submission_filepath}" ]; then
  echo "Generate test_prediction.csv from ${submission_filepath}"
  cp ${submission_filepath} test_prediction.csv
  zip test_prediction.zip test_prediction.csv
  rm test_prediction.csv
else
  echo "${submission_filepath} not exists"
fi
