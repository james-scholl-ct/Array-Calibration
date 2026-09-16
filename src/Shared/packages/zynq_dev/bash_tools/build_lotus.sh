#!/usr/bin/bash

git_root=`git rev-parse --show-toplevel | sed -e 's/^C:/\/c/'`
build_dir=$git_root/_build_lotus

rm -rf $build_dir
mkdir $build_dir
pushd $build_dir > /dev/null

# aliases to run vivado tcl script
#   * mode=batch returns to bash on either `exit`, `return`, or EOF)
#   * mode=tcl returns to bash on `exit` and to tclsh on either `return` or EOF
alias vivado_sh="vivado -mode batch -source "
alias vivado_tcl="vivado -mode tcl -source "

# vivado_sh $git_root/tcl/lotus.tcl -tclargs --help
# vivado_tcl $git_root/tcl/lotus.tcl
vivado_sh $git_root/tcl/lotus.tcl

echo "Bitsteam built here: `pwd`"


echo vivado.log > logfiles
#find ../vivado_projects/lotus -name '*.vds' >> logfiles
#find ../vivado_projects/lotus -name '*.vdi' >> logfiles

#echo ../vivado_projects/lotus/lotus.runs/lotus_auto_pc_0_synth_1/lotus_auto_pc_0.vds >> logfiles
#echo ../vivado_projects/lotus/lotus.runs/lotus_clk_wiz_0_0_synth_1/lotus_clk_wiz_0_0.vds >> logfiles
#echo ../vivado_projects/lotus/lotus.runs/lotus_xbar_0_synth_1/lotus_xbar_0.vds >> logfiles
echo ../vivado_projects/lotus/lotus.runs/lotus_lotus_bsc_0_0_synth_1/lotus_lotus_bsc_0_0.vds >> logfiles
echo ../vivado_projects/lotus/lotus.runs/lotus_lotus_spi_0_0_synth_1/lotus_lotus_spi_0_0.vds >> logfiles
#echo ../vivado_projects/lotus/lotus.runs/lotus_processing_system7_0_0_synth_1/lotus_processing_system7_0_0.vds >> logfiles
#echo ../vivado_projects/lotus/lotus.runs/lotus_rst_ps7_0_100M_0_synth_1/lotus_rst_ps7_0_100M_0.vds >> logfiles
echo ../vivado_projects/lotus/lotus.runs/synth_1/lotus_wrapper.vds >> logfiles

echo ../vivado_projects/lotus/lotus.runs/impl_1/lotus_wrapper.vdi >> logfiles


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
