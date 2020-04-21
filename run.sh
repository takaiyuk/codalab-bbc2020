#!/bin/bash

if [ "$1" == "" ]; then
  make data && make features && make models && make submit
elif [ "$1" == "data" ]; then
  make data && make features && make models && make submit
elif [ "$1" == "features" ]; then
  make features && make models && make submit
elif [ "$1" == "models" ]; then
  make models && make submit
elif [ "$1" == "submit" ]; then
  make submit
else
  echo "Error: arguments must be one of (data, features, models, submit)"
  exit 1
fi
