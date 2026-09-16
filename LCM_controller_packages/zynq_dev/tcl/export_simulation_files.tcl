# export simulation sources

set cant_export_sim 0
if { ! [info exists project_name] } {
    puts "ERROR: variable 'project_name' is not defined"
    set cant_export_sim 1
}
if { ! [info exists project_path] } {
    puts "ERROR: variable 'project_path' is not defined"
    set cant_export_sim 1
}

if { $cant_export_sim } {
    puts "ERROR: can't export simulation files due to earlier errors"
    return 0
}

set sim_export_dir _sim_export_${project_name}
file delete -force $sim_export_dir

export_ip_user_files -no_script -force
export_simulation  \
    -simulator xsim  \
    -directory $sim_export_dir \
    -ip_user_files_dir $project_path/${project_name}.ip_user_files \
    -ipstatic_source_dir $project_path/${project_name}.ip_user_files/ipstatic \
    -use_ip_compiled_libs \
    -export_source_files
