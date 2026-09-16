# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
set project_name orchid
set project_root vivado_projects
set ip_dir ip_repo

set git_root [file normalize [exec git rev-parse --show-toplevel]]
set project_path [file normalize $git_root/$project_root/$project_name]
set part {xc7z010clg400-1}
set top_wrapper_name ${project_name}_wrapper
set top_tb_name tb_${project_name}
set top_constraints_file ${project_name}_constraints.xdc
set bd_tcl_file_path $git_root/tcl/${project_name}_bd.tcl
set bitstream_path [file normalize ./$project_name.bit]
set message_config_path $git_root/tcl/${project_name}_message_config.tcl

set ip_project_name ${project_name}_ip
set ip_project_path [file normalize ${project_path}_ip]
set ip_dir_path $git_root/$project_root/$ip_dir/$project_name
set ip_message_config_path $git_root/tcl/${ip_project_name}_message_config.tcl

# -----------------------------------------------------------------------------
# Command line arguments
# -----------------------------------------------------------------------------
proc help {} {
    set script_file [info script]
    puts ""
    puts "Description:"
    puts "    Build Orchid."
    puts ""
    puts "Usage:"
    puts "    # vivado is C:\\Xilinx\\Vivado\\2018.2\\bin\\vivado.bat"
    puts "    vivado -mode batch -source $script_file \[-tclargs <args>\]"
    puts ""
    puts "Args:"
    puts "--------------------------------------------------------------------"
    puts "\[--project_root <name>\]  The directory in which to create the"
    puts "                         vivado projects, relative to git root."
    puts "                         Default is vivado_projects.\n"
    puts "\[--project_name <name>\]  Create project with the specified name."
    puts "                         Default is [file rootname $script_file].\n"
    puts "\[--help\]                 Print help information for this script."
    puts "--------------------------------------------------------------------"
    puts ""
}

if { $::argc > 0 } {
    for {set i 0} {$i < $::argc} {incr i} {
        set option [string trim [lindex $::argv $i]]
        switch -regexp {--} $option {
            "--project_root" { incr i; set project_root [lindex $::argv $i] }
            "--project_name" { incr i; set project_name [lindex $::argv $i] }
            "--help"         { help; exit 0 }
            default {
                if { [regexp {^-} $option] } {
                    puts "ERROR: Unknown option '$option' specified.\n"
                    help
                    exit 1
                }
            }
        }
    }
}

# -----------------------------------------------------------------------------
# Create project
# -----------------------------------------------------------------------------
create_project -force -part $part $project_name $project_path
set_property board_part em.avnet.com:microzed_7010:part0:1.1 [current_project]
set board_id microzed_7010

# Apply message rules
source $message_config_path

# Set project properties
set obj [current_project]
set_property -name "default_lib" -value "xil_defaultlib" -objects $obj
set_property -name "dsa.accelerator_binary_content" -value "bitstream" -objects $obj
set_property -name "dsa.accelerator_binary_format" -value "xclbin2" -objects $obj
set_property -name "dsa.board_id" -value $board_id -objects $obj
set_property -name "dsa.description" -value "Vivado generated DSA" -objects $obj
set_property -name "dsa.dr_bd_base_address" -value "0" -objects $obj
set_property -name "dsa.emu_dir" -value "emu" -objects $obj
set_property -name "dsa.flash_interface_type" -value "bpix16" -objects $obj
set_property -name "dsa.flash_offset_address" -value "0" -objects $obj
set_property -name "dsa.flash_size" -value "1024" -objects $obj
set_property -name "dsa.host_architecture" -value "x86_64" -objects $obj
set_property -name "dsa.host_interface" -value "pcie" -objects $obj
set_property -name "dsa.num_compute_units" -value "60" -objects $obj
set_property -name "dsa.platform_state" -value "pre_synth" -objects $obj
set_property -name "dsa.uses_pr" -value "1" -objects $obj
set_property -name "dsa.vendor" -value "xilinx" -objects $obj
set_property -name "dsa.version" -value "0.0" -objects $obj
set_property -name "enable_vhdl_2008" -value "1" -objects $obj
set_property -name "ip_cache_permissions" -value "read write" -objects $obj
set_property -name "ip_output_repo" -value "$project_path/${project_name}.cache/ip" -objects $obj
set_property -name "mem.enable_memory_map_generation" -value "1" -objects $obj
set_property -name "sim.central_dir" -value "$project_path/${project_name}.ip_user_files" -objects $obj
set_property -name "sim.ip.auto_export_scripts" -value "1" -objects $obj
set_property -name "simulator_language" -value "Mixed" -objects $obj
set_property -name "target_language" -value "VHDL" -objects $obj
set_property -name "webtalk.activehdl_export_sim" -value "11" -objects $obj
set_property -name "webtalk.ies_export_sim" -value "11" -objects $obj
set_property -name "webtalk.modelsim_export_sim" -value "11" -objects $obj
set_property -name "webtalk.questa_export_sim" -value "11" -objects $obj
set_property -name "webtalk.riviera_export_sim" -value "11" -objects $obj
set_property -name "webtalk.vcs_export_sim" -value "11" -objects $obj
set_property -name "webtalk.xsim_export_sim" -value "11" -objects $obj
set_property -name "xpm_libraries" -value "XPM_CDC XPM_FIFO XPM_MEMORY" -objects $obj

# -----------------------------------------------------------------------------
# IP sub-project
# -----------------------------------------------------------------------------
create_project -force -part $part $ip_project_name $ip_project_path
set_property board_part em.avnet.com:microzed_7010:part0:1.1 [current_project]

# Apply message rules
source $ip_message_config_path

#set tmp [add_files -norecurse [list  ]]
#set_property file_type {Verilog Header} [get_files $tmp]

#set tmp [add_files -norecurse [list  ]]
#set_property file_type {Memory Initialization Files} [get_files $tmp]

set files [glob $git_root/rtl_src/${project_name}*.v]
#add_files -norecurse $files
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

# -----------------------------------------------------------------------------
# Configure sources
# -----------------------------------------------------------------------------
# Create 'sources_1' fileset
if {[string equal [get_filesets -quiet sources_1] ""]} {
    create_fileset -srcset sources_1
}

# Set IP repository paths
set obj [get_filesets sources_1]
set_property ip_repo_paths $ip_dir_path $obj

# Rebuild user ip_repo's index before adding any source files
update_ip_catalog -rebuild

# Set 'sources_1' fileset object
# Empty (no sources present)

# Set 'sources_1' fileset properties
set_property -name top -value $top_wrapper_name -objects $obj

# -----------------------------------------------------------------------------
# Configure constraints
# -----------------------------------------------------------------------------
# Create 'constrs_1' fileset
if {[string equal [get_filesets -quiet constrs_1] ""]} {
    create_fileset -constrset constrs_1
}

# Set 'constrs_1' fileset object
set obj [get_filesets constrs_1]
set files [list \
    $git_root/constraints/$top_constraints_file]
add_files -fileset constrs_1 -norecurse $files
set file_obj [get_files -of_objects $obj]
set_property -name file_type -value XDC -objects $file_obj

# Set 'constrs_1' fileset properties
set_property -name target_constrs_file -value [get_files $top_constraints_file] -objects $obj
set_property -name target_ucf -value [get_files $top_constraints_file] -objects $obj
set_property -name target_part -value $part -objects $obj

# -----------------------------------------------------------------------------
# Configure simulations
# -----------------------------------------------------------------------------
# Create 'sim_1' fileset
if {[string equal [get_filesets -quiet sim_1] ""]} {
    create_fileset -simset sim_1
}

# Set 'sim_1' fileset object
set obj [get_filesets sim_1]
set files [concat \
    [glob $git_root/sim_src/$top_tb_name*.v] \
    [glob $git_root/sim_src/$top_tb_name*.wcfg]]
add_files -fileset sim_1 -norecurse $files

# Set 'sim_1' fileset properties
set obj [get_filesets sim_1]
set_property -name top -value $top_tb_name -objects $obj
set_property -name top_auto_set -value 0 -objects $obj

# -----------------------------------------------------------------------------
# Create block design
# -----------------------------------------------------------------------------
source $bd_tcl_file_path
set bd_file [get_files ${project_name}.bd]
set wrapper [make_wrapper -files $bd_file -top]
add_files -norecurse $wrapper
update_compile_order -fileset sources_1
update_compile_order -fileset sim_1

# Generate target data for block design. The target data that are generated
# are the files necessary to support the block design through the FPGA design
# flow (simulation, synthesis, and implementation).
generate_target all $bd_file

# -----------------------------------------------------------------------------
# Export simulation objects for behavioral simulations
# -----------------------------------------------------------------------------
source $git_root/tcl/export_simulation_files.tcl

# -----------------------------------------------------------------------------
# Configure synthesis
# -----------------------------------------------------------------------------
# Create 'synth_1' run
if {[string equal [get_runs synth_1] ""]} {
    create_run -name synth_1 -flow {Vivado Synthesis 2018}
}
set obj [get_runs synth_1]
set_property -name part -value $part -objects $obj
set_property -name strategy -value {Vivado Synthesis Defaults} -objects $obj
set_property -name constrset -value constrs_1 -objects $obj
set_property -name report_strategy -value {Vivado Synthesis Default Reports} -objects $obj

# set the current synth run
current_run -synthesis [get_runs synth_1]

# -----------------------------------------------------------------------------
# Configure implementation
# -----------------------------------------------------------------------------
# Create 'impl_1' run
if {[string equal [get_runs impl_1] ""]} {
    create_run -name impl_1 -flow {Vivado Implementation 2018}
}
set obj [get_runs impl_1]
set_property -name part -value $part -objects $obj
set_property -name strategy -value {Vivado Implementation Defaults} -objects $obj
set_property -name constrset -value constrs_1 -objects $obj
set_property -name report_strategy -value {Vivado Implementation Default Reports} -objects $obj

# set the current impl run
current_run -implementation [get_runs impl_1]

# -----------------------------------------------------------------------------
# Run synthesis, implementation, and bitstream generation
# -----------------------------------------------------------------------------
# Run synthesis
launch_runs synth_1 -jobs 2
wait_on_run synth_1

# Run implementation
launch_runs impl_1 -to_step write_bitstream -jobs 2
wait_on_run impl_1

# Copy bitstream
file copy -force $project_path/$project_name.runs/impl_1/$top_wrapper_name.bit $bitstream_path
puts "INFO: Bitstream created for project ${project_name}: $bitstream_path"
