# create and package ip

create_project -force -part $part $ip_project_name $ip_project_path
set_property board_part em.avnet.com:microzed_7010:part0:1.1 [current_project]

source $ip_message_config_path

set tmp [add_files -norecurse [list $git_root/rtl_src/${project_name}_defines.v]]
set_property file_type {Verilog Header} [get_files $tmp]

#set files [glob $git_root/rtl_src/${project_name}*.memh]
#set tmp [add_files -norecurse $files]
#set_property file_type {Memory Initialization Files} [get_files $tmp]

set files [list \
    $git_root/rtl_src/lotus.v \
    $git_root/rtl_src/lotus_active_mem.v \
    $git_root/rtl_src/lotus_axi_interface.v \
    $git_root/rtl_src/lotus_core.v \
    $git_root/rtl_src/lotus_data_mux.v \
    $git_root/rtl_src/lotus_fsm_lvds_data.v \
    $git_root/rtl_src/lotus_fsm_scan_seq.v \
    $git_root/rtl_src/lotus_laser_control.v \
    $git_root/rtl_src/lotus_output_drivers.v \
    $git_root/rtl_src/lotus_rom.v \
    $git_root/rtl_src/lotus_sync_a2b.v \
    $git_root/rtl_src/lotus_table_mover.v \
]

read_verilog $files
update_compile_order -fileset sources_1

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
