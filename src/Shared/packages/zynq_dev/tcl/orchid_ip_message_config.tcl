# don't import files into project because they are managed in version control. (UG1198)
set_msg_config -new_severity INFO -id {IP_Flow 19-459} -string {{CRITICAL WARNING: [IP_Flow 19-459] IP file}}

# localparam used for port size
set_msg_config -new_severity INFO -id {IP_Flow 19-587} -string {{WARNING: [IP_Flow 19-587] [HDL Parser] HDL port or parameter 'axi_memory' has a dependency on the module local parameter or undefined parameter 'MEMORY_N_BITS'.}}

# These don't appear to affect anything
set_msg_config -new_severity INFO -id {IP_Flow 19-3153} -string {{WARNING: [IP_Flow 19-3153] Bus Interface 'LVDS_CLK_N': ASSOCIATED_BUSIF bus parameter is missing.}}
set_msg_config -new_severity INFO -id {IP_Flow 19-3153} -string {{WARNING: [IP_Flow 19-3153] Bus Interface 'LVDS_CLK_P': ASSOCIATED_BUSIF bus parameter is missing.}}

set_msg_config -new_severity INFO -id {IP_Flow 19-4751} -string {{WARNING: [IP_Flow 19-4751] Bus Interface 'LVDS_CLK_N': FREQ_HZ bus parameter is missing for output clock interface.}}
set_msg_config -new_severity INFO -id {IP_Flow 19-4751} -string {{WARNING: [IP_Flow 19-4751] Bus Interface 'LVDS_CLK_P': FREQ_HZ bus parameter is missing for output clock interface.}}
