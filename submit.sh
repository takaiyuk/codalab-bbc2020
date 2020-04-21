#!/bin/bash

# Generate test_prediction.csv for submission from submit/${session_id}.csv
ls_submit=$(ls -r submit)
submits=(${ls_submit// / })
submission_filepath=${submits[0]}

echo "Generate test_prediction.csv from submit/${submission_filepath}"
cp submit/${submission_filepath} test_prediction.csv
zip test_prediction.zip test_prediction.csv
