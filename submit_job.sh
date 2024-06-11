#!/bin/bash

# Define the output directory
output_dir="/pbs_outputs/$1"

# Print the output directory path
echo "$output_dir"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"
