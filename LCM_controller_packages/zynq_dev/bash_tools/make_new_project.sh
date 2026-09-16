#!/usr/bin/bash

# Make a new project by copying and renaming sources from the genesis project
#
# ${parameter/pattern/string} replaces pattern with string in a variable
# See: https://www.gnu.org/software/bash/manual/bash.html#Shell-Parameter-Expansion
#


if [ ! $# -eq 1 ]; then
    echo "Please provide the name of the design on the command line"
    return
fi
design=$1


git_root=`git rev-parse --show-toplevel | sed -e 's/^C:/\/c/'`
log=$git_root/new_project_files.txt
rm -f $log


# rtl_src
pushd $git_root/rtl_src
for f in genesis*.v; do
    cp -- "$f" "${f/genesis/$design}"
done
perl -p -i.bak -e "s/genesis/$design/g" $design*.v
rm $design*.v.bak
for f in $design*.v; do echo rtl_src/$f >> $log; done
popd


# sim_src
pushd $git_root/sim_src
for f in tb_genesis*.*; do
    cp -- "$f" "${f/genesis/$design}"
done
perl -p -i.bak -e "s/genesis/$design/g" tb_${design}*.*
rm tb_${design}*.*.bak
perl -p -i.bak -e "s/GENESIS/uc($design)/eg" tb_${design}*.*
rm tb_${design}*.*.bak
for f in tb_${design}*.*; do echo sim_src/$f >> $log; done
popd


# constraints
pushd $git_root/constraints
for f in genesis*.xdc; do
    cp -- "$f" "${f/genesis/$design}"
done
for f in ${design}*.xdc; do echo constraints/$f >> $log; done
popd


# tcl
pushd $git_root/tcl
for f in genesis*.tcl; do
    cp -- "$f" "${f/genesis/$design}"
done
perl -p -i.bak -e "s/genesis/$design/g" ${design}*.tcl
rm ${design}*.tcl.bak
for f in ${design}*.tcl; do echo tcl/$f >> $log; done
popd


# bash_tools
pushd $git_root/bash_tools
for f in build_genesis.sh; do
    cp -- "$f" "${f/genesis/$design}"
done
perl -p -i.bak -e "s/genesis/$design/g" build_${design}.sh
rm build_${design}.sh.bak
for f in build_${design}.sh; do echo bash_tools/$f >> $log; done
popd


# apps
pushd $git_root/apps
cp -rT genesis $design
cd $design
for f in genesis*.*; do
    mv -- "$f" "${f/genesis/$design}"
done
perl -p -i.bak -e "s/genesis/$design/g" *
rm *.bak
perl -p -i.bak -e "s/GENESIS/uc($design)/eg" *
rm *.bak
for f in *; do echo apps/$design/$f >> $log; done
popd


# unittests
pushd $git_root/unittests
for f in test_genesis.py; do
    cp -- "$f" "${f/genesis/$design}"
done
perl -p -i.bak -e "s/genesis/$design/g" test_${design}.py
rm test_${design}.py.bak
for f in test_${design}.py; do echo unittests/$f >> $log; done
popd


# python_tools
pushd $git_root/python_tools
for f in genesis_zynq_api.py; do
    cp -- "$f" "${f/genesis/$design}"
done
perl -p -i.bak -e "s/genesis/$design/g" ${design}_zynq_api.py
rm ${design}_zynq_api.py.bak
perl -p -i.bak -e "s/Genesis/\u$design/g" ${design}_zynq_api.py
rm ${design}_zynq_api.py.bak
for f in ${design}_zynq_api.py; do echo python_tools/$f >> $log; done
popd


# python_tools/unittests
pushd $git_root/python_tools/unittests
for f in test_genesis_zynq_api.py; do
    cp -- "$f" "${f/genesis/$design}"
done
perl -p -i.bak -e "s/genesis/$design/g" test_${design}_zynq_api.py
rm test_${design}_zynq_api.py.bak
perl -p -i.bak -e "s/Genesis/\u$design/g" test_${design}_zynq_api.py
rm test_${design}_zynq_api.py.bak
for f in test_${design}_zynq_api.py; do echo python_tools/unittests/$f >> $log; done
popd


cat $log | sort > $log.bak
mv $log.bak $log


echo
echo "Done. Project files are listed in $log."
echo "To add the files, run: cat $log | xargs git add" 
