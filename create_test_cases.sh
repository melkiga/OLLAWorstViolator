#!/bin/bash

EXECUTABLE_NAME=osvm
TYPE=Release
RUN_FILE_PATH=bin/$TYPE/$EXECUTABLE_NAME

[ ! -f "$RUN_FILE_PATH" ] && echo "Error: ${RUN_FILE_PATH} does not exist." && exit 1

datasets=$(ls small-data/iris small-data/sonar small-data/teach small-data/pro small-data/vote)

for dataset in $datasets; do
    $RUN_FILE_PATH -i 5 -o 5 -M 0.001 $dataset
done


