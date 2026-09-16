#!/bin/bash

prog=orchid_driver

git_root=`git rev-parse --show-toplevel`
rm -f run.elf
echo "Compiling sources"
gcc -Wall -lm -I.. -o run.elf ../${prog}.c test_${prog}.c

echo "Programming FPGA"
cat ~/orchid/orchid.bit > /dev/xdevcfg
sleep 1
status=$(cat /sys/class/xdevcfg/xdevcfg/device/prog_done)
if [ $status -ne 1 ]; then
    echo "Failed to program FPGA"
    return
fi

echo "Running tests"
echo "--------------------------------"
./run.elf


