#!/usr/bin/bash

# Parse args
if [ $# -eq 1 ]; then
    if [[ $1 = sim_only ]]; then
        tclargs="-tclargs --sim_only"
    else
        tclargs=""
    fi
else
    tclargs=""
fi

git_root=`git rev-parse --show-toplevel | sed -e 's/^C:/\/c/'`
build_dir=$git_root/_build_genesis
tcl_source=$git_root/tcl/genesis.tcl

rm -rf $build_dir
mkdir $build_dir
pushd $build_dir > /dev/null

# aliases to run vivado tcl script
#   * mode=batch returns to bash on either `exit`, `return`, or EOF)
#   * mode=tcl returns to bash on `exit` and to tclsh on either `return` or EOF
alias vivado_sh="vivado -mode batch -source $tcl_source $tclargs"
alias vivado_tcl="vivado -mode tcl -source $tcl_source $tclargs"

echo -n "Running: "
echo $(alias vivado_sh)
vivado_sh
echo
echo "Bitstream built here: `pwd`"


echo vivado.log > logfiles
#find ../vivado_projects/genesis -name '*.vds' >> logfiles
#find ../vivado_projects/genesis -name '*.vdi' >> logfiles

#echo ../vivado_projects/genesis/genesis.runs/genesis_auto_pc_0_synth_1/genesis_auto_pc_0.vds >> logfiles
#echo ../vivado_projects/genesis/genesis.runs/genesis_xbar_0_synth_1/genesis_xbar_0.vds >> logfiles
echo ../vivado_projects/genesis/genesis.runs/genesis_genesis_ip0_0_0_synth_1/genesis_genesis_ip0_0_0.vds >> logfiles
echo ../vivado_projects/genesis/genesis.runs/genesis_genesis_ip1_0_0_synth_1/genesis_genesis_ip1_0_0.vds >> logfiles
#echo ../vivado_projects/genesis/genesis.runs/genesis_processing_system7_0_0_synth_1/genesis_processing_system7_0_0.vds >> logfiles
#echo ../vivado_projects/genesis/genesis.runs/genesis_rst_ps7_0_100M_0_synth_1/genesis_rst_ps7_0_100M_0.vds >> logfiles
echo ../vivado_projects/genesis/genesis.runs/synth_1/genesis_wrapper.vds >> logfiles

echo ../vivado_projects/genesis/genesis.runs/impl_1/genesis_wrapper.vdi >> logfiles


cat `cat logfiles` > all.log

grep '^\(WARNING\|CRITICAL WARNING\|ERROR\):' all.log > issues.log
issue_cnt=`wc -l < issues.log`
echo "There are $issue_cnt issues. Please see `pwd`/issues.log"

popd > /dev/null

# Save version information
pushd $git_root
touch git.log
cmd="git branch -vv"
echo "> $cmd" >> git.log
eval $cmd >> git.log
echo >> git.log
cmd="git clean -dnx"
echo "> $cmd" >> git.log
eval $cmd >> git.log
echo >> git.log
cmd="git status"
echo "> $cmd" >> git.log
eval $cmd >> git.log

mv git.log $build_dir/
popd > /dev/null
