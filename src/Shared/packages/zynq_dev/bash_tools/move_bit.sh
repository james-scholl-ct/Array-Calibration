#!/usr/bin/bash

fpga_host=root@cube-10-12-15-cube-server
fpga_git_root=/root/git_repos/zynq_dev

if [ ! $# -eq 1 ]; then
    echo "Please provide the name of the module on the command line"
    return
fi

src=$git_root/_build_${1}/${1}.bit
dst=$fpga_git_root/${1}.bit

git_root=`git rev-parse --show-toplevel | sed -e 's/^C:/\/c/'`
if [ ! -f $src ]; then
    echo "file not found $src"
    return
fi

scp $src $fpga_host:$dst
ssh $fpga_host "cat $dst > /dev/xdevcfg"
echo
echo "Checking status: ssh $fpga_host 'cat /sys/class/xdevcfg/xdevcfg/device/prog_done'"
ssh $fpga_host "cat /sys/class/xdevcfg/xdevcfg/device/prog_done"
