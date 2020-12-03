#!/bin/bash

# Initial Setup
source /data/intel_fpga/devcloudLoginToolSetup.sh
tools_setup -t A10DS

# Move to project directory
cd ~/lab1/

# Running project in Emulation mode
printf "\\n%s\\n" "Running in Emulation Mode:"
aoc -march=emulator -v device/cnn.cl -o bin/cnn.aocx
make

# Run host code for version 1.2.1
./bin/host -emulator
