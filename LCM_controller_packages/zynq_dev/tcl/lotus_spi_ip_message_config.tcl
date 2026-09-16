# don't import files into project because they are managed in version control. (UG1198)
set_msg_config -new_severity INFO -id {IP_Flow 19-459} -string {{CRITICAL WARNING: [IP_Flow 19-459] IP file}}

# localparam used for port size
set_msg_config -new_severity INFO -id {IP_Flow 19-587} -string {{WARNING: [IP_Flow 19-587] [HDL Parser] HDL port or parameter}}

# global include file is included multiple times
set_msg_config -new_severity INFO -id {IP_Flow 19-4816} -string {{WARNING: [IP_Flow 19-4816] The Synthesis file group has two include files that have the same base name. It is not guaranteed which of these two files will be picked up during synthesis/simulation:   ../../../rtl_src/lotus_defines.v}}
set_msg_config -new_severity INFO -id {IP_Flow 19-4816} -string {{WARNING: [IP_Flow 19-4816] The Simulation file group has two include files that have the same base name. It is not guaranteed which of these two files will be picked up during synthesis/simulation:   ../../../rtl_src/lotus_defines.v}}

# ignore project management stuff
set_msg_config -new_severity INFO -id {IP_Flow 19-731} -string {{WARNING: [IP_Flow 19-731] File Group 'xilinx_anylanguagesynthesis (Synthesis)': } {file path is not relative to the IP root directory.}}
set_msg_config -new_severity INFO -id {IP_Flow 19-731} -string {{WARNING: [IP_Flow 19-731] File Group 'xilinx_anylanguagebehavioralsimulation (Simulation)': } {file path is not relative to the IP root directory.}}
