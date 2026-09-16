# don't import files into project because they are managed in version control. (UG1198)
set_msg_config -new_severity INFO -id {IP_Flow 19-459} -string {{CRITICAL WARNING: [IP_Flow 19-459] IP file}}

# localparam used for port size
set_msg_config -new_severity INFO -id {IP_Flow 19-587} -string {{WARNING: [IP_Flow 19-587] [HDL Parser] HDL port or parameter}}
