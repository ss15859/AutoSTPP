#!/bin/bash

# for each dirname in output_data run "make run_stpp_earthquakeNPP config=dirname"
for dirname in $(ls output_data); do
    # check if the directory exists
    if [ -d "output_data/$dirname" ]; then
        # run the make command
        echo "Running make run_stpp_earthquakeNPP config=$dirname"
        make run_stpp_earthquakeNPP config=$dirname
    else
        echo "Directory output_data/$dirname does not exist."
    fi
done
# for dirname in $(ls output_data); do