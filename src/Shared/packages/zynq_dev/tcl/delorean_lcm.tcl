# create and package ip

create_project -force -part $part $ip_project_name $ip_project_path
set_property board_part $board_part [current_project]

source $ip_message_config_path

set tmp [add_files -norecurse [list $git_root/rtl_src/${project_name}_defines.v]]
set_property file_type {Verilog Header} [get_files $tmp]

set files [list \
    $git_root/rtl_src/delorean_lcm.v \
    $git_root/rtl_src/delorean_regs.v \
    $git_root/rtl_src/delorean_axi_interface.v \
    $git_root/rtl_src/delorean_dma_bram.v \
    $git_root/rtl_src/delorean_mlvds.v \
    $git_root/rtl_src/delorean_tcon.v \
    $git_root/rtl_src/delorean_laser_control.v \
    $git_root/rtl_src/delorean_lcm_io_mlvds.v \
    $git_root/rtl_src/delorean_lcm_io_laser.v \
]

read_verilog $files
update_compile_order -fileset sources_1

# elaborate and open RTL design
synth_design -rtl -name rtl_1

ipx::package_project \
    -root_dir $ip_dir_path \
    -vendor xilinx.com \
    -library ip \
    -taxonomy /UserIP \
    -force

# Save and close the packaging project
set_property core_revision 2 [ipx::current_core]
ipx::update_checksums [ipx::current_core]
ipx::save_core [ipx::current_core]
close_project
