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
build_dir=$git_root/_build_bcuda
tcl_source=$git_root/tcl/bcuda.tcl

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
#find ../vivado_projects/bcuda -name '*.vds' >> logfiles
#find ../vivado_projects/bcuda -name '*.vdi' >> logfiles

#echo ../vivado_projects/bcuda/bcuda.runs/bcuda_auto_pc_0_synth_1/bcuda_auto_pc_0.vds >> logfiles
#echo ../vivado_projects/bcuda/bcuda.runs/bcuda_xbar_0_synth_1/bcuda_xbar_0.vds >> logfiles
echo ../vivado_projects/bcuda/bcuda.runs/bcuda_bcuda_ip0_0_0_synth_1/bcuda_bcuda_ip0_0_0.vds >> logfiles
echo ../vivado_projects/bcuda/bcuda.runs/bcuda_bcuda_ip1_0_0_synth_1/bcuda_bcuda_ip1_0_0.vds >> logfiles
#echo ../vivado_projects/bcuda/bcuda.runs/bcuda_processing_system7_0_0_synth_1/bcuda_processing_system7_0_0.vds >> logfiles
#echo ../vivado_projects/bcuda/bcuda.runs/bcuda_rst_ps7_0_0_synth_1/bcuda_rst_ps7_0_0.vds >> logfiles
echo ../vivado_projects/bcuda/bcuda.runs/synth_1/bcuda_wrapper.vds >> logfiles

echo ../vivado_projects/bcuda/bcuda.runs/impl_1/bcuda_wrapper.vdi >> logfiles


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
