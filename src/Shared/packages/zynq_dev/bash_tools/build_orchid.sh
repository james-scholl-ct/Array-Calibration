#!/usr/bin/bash

git_root=`git rev-parse --show-toplevel | sed -e 's/^C:/\/c/'`
build_dir=$git_root/_build_orchid

rm -rf $build_dir
mkdir $build_dir
pushd $build_dir > /dev/null

# aliases to run vivado tcl script
#   * mode=batch returns to bash on either `exit`, `return`, or EOF)
#   * mode=tcl returns to bash on `exit` and to tclsh on either `return` or EOF
alias vivado_sh="vivado -mode batch -source "
alias vivado_tcl="vivado -mode tcl -source "

# vivado_sh $git_root/tcl/orchid.tcl -tclargs --help
# vivado_tcl $git_root/tcl/orchid.tcl
vivado_sh $git_root/tcl/orchid.tcl

echo "Bitsteam built here: `pwd`"


echo vivado.log > logfiles
#find ../vivado_projects/orchid -name '*.vds' >> logfiles
#find ../vivado_projects/orchid -name '*.vdi' >> logfiles

#echo ../vivado_projects/orchid/orchid.runs/orchid_auto_pc_0_synth_1/orchid_auto_pc_0.vds >> logfiles
#echo ../vivado_projects/orchid/orchid.runs/orchid_clk_wiz_0_0_synth_1/orchid_clk_wiz_0_0.vds >> logfiles
echo ../vivado_projects/orchid/orchid.runs/orchid_orchid_0_0_synth_1/orchid_orchid_0_0.vds >> logfiles
#echo ../vivado_projects/orchid/orchid.runs/orchid_processing_system7_0_0_synth_1/orchid_processing_system7_0_0.vds >> logfiles
#echo ../vivado_projects/orchid/orchid.runs/orchid_rst_ps7_0_100M_0_synth_1/orchid_rst_ps7_0_100M_0.vds >> logfiles
echo ../vivado_projects/orchid/orchid.runs/synth_1/orchid_wrapper.vds >> logfiles

echo ../vivado_projects/orchid/orchid.runs/impl_1/orchid_wrapper.vdi >> logfiles


cat `cat logfiles` > all.log

grep '^\(WARNING\|CRITICAL WARNING\|ERROR\):' all.log > issues.log
issue_cnt=`wc -l < issues.log`
echo "There are $issue_cnt issues. Please see `pwd`/issues.log" 

popd > /dev/null
