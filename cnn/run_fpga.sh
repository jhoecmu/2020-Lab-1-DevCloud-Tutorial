#!/bin/bash

# Initial Setup
source /data/intel_fpga/devcloudLoginToolSetup.sh
tools_setup -t A10DS

# Move to project directory
cd ~/lab1/

# Check Arria 10 PAC card connectivity
aocl diagnose
error_check

# Running project in FPGA Hardware Mode
printf "\\n%s\\n" "Running in FPGA Hardware Mode:"
aoc -list-boards
error_check
# Get device name
aocl diagnose
error_check

# Programmming PAC Card
aocl program acl0 bin/cnn_unsigned.aocx
./bin/host
error_check
