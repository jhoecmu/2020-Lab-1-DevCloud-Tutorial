# Initial Setup
source /data/intel_fpga/devcloudLoginToolSetup.sh
tools_setup -t A10DS

# Move to project directory
cd ~/lab1/

# Check Arria 10 PAC card connectivity
aocl diagnose
error_check

# Running project in FPGA Hardware Mode (this takes approximately 1 hour)
printf "\\n%s\\n" "Running in FPGA Hardware Mode:"
aoc device/cnn.cl -o bin/cnn.aocx -board=pac_a10

# Availability of Acceleration cards
aoc -list-boards
error_check
# Get device name
aocl diagnose
error_check

# Converting to an unsigned .aocx file
cd bin
printf "\\n%s\\n" "Converting to unsigned .aocx:"
printf "Y\\nY\\n" | source $AOCL_BOARD_PACKAGE_ROOT/linux64/libexec/sign_aocx.sh -H openssl_manager -i cnn.aocx -r NULL -k NULL -o cnn_unsigned.aocx
error_check
# Programmming PAC Card
aocl program acl0 cnn_unsigned.aocx
./host
error_check