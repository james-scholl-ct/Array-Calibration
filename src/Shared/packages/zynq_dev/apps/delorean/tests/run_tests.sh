#!/bin/bash

echo "Programming FPGA"
echo "--------------------------------"
cat ~/delorean/delorean.bit > /dev/xdevcfg
sleep 2
status=$(cat /sys/class/xdevcfg/xdevcfg/device/prog_done)
if [ $status -ne 1 ]; then
    echo "Failed to program FPGA."
    return
else
    echo "Configuration succeeded."
fi

echo
echo "Compiling sources"
echo "--------------------------------"
make clean
make
sleep 2

echo
echo "Running tests"
echo "--------------------------------"
for elf in *.elf; do
    ./$elf 0
done
