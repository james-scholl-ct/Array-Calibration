# don't import files into project because they are managed in version control. (UG1198)
set_msg_config -new_severity INFO -id {IP_Flow 19-459} -string {{CRITICAL WARNING: [IP_Flow 19-459] IP file}}

# localparam used for port size
set_msg_config -new_severity INFO -id {IP_Flow 19-587} -string {{WARNING: [IP_Flow 19-587] [HDL Parser] HDL port or parameter}}

# global include file is included multiple times
set_msg_config -new_severity INFO -id {IP_Flow 19-4816} -string {{WARNING: [IP_Flow 19-4816] The Synthesis file group has two include files that have the same base name. It is not guaranteed which of these two files will be picked up during synthesis/simulation:   ../../../rtl_src/delorean_defines.v}}
set_msg_config -new_severity INFO -id {IP_Flow 19-4816} -string {{WARNING: [IP_Flow 19-4816] The Simulation file group has two include files that have the same base name. It is not guaranteed which of these two files will be picked up during synthesis/simulation:   ../../../rtl_src/delorean_defines.v}}

# ignore project management stuff
set_msg_config -new_severity INFO -id {IP_Flow 19-731} -string {{WARNING: [IP_Flow 19-731] File Group 'xilinx_anylanguagesynthesis (Synthesis)': } {file path is not relative to the IP root directory.}}
set_msg_config -new_severity INFO -id {IP_Flow 19-731} -string {{WARNING: [IP_Flow 19-731] File Group 'xilinx_anylanguagebehavioralsimulation (Simulation)': } {file path is not relative to the IP root directory.}}

# our code doesn't use this so must be an automatic step in synth_design
set_msg_config -new_severity INFO -id {Common 17-576} -string {{WARNING: [Common 17-576] 'use_project_ipc' is deprecated. This option is deprecated and no longer used.}}

# -----------------------------------------------------------------------------
# Approved elaboration/synthesis messages
# -----------------------------------------------------------------------------
# replaced by macro
set_msg_config -new_severity INFO -id {Synth 8-6014} -string {{WARNING: [Synth 8-6014] Unused sequential element inst/U_delorean_dma_bram/ram_reg was removed.}}

# Unused ports
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_regs has unconnected port CH0_RD_ADDR[1]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_regs has unconnected port CH0_RD_ADDR[0]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_axi_interface has unconnected port S_AXI_AWPROT[2]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_axi_interface has unconnected port S_AXI_AWPROT[1]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_axi_interface has unconnected port S_AXI_AWPROT[0]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_axi_interface has unconnected port S_AXI_ARPROT[2]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_axi_interface has unconnected port S_AXI_ARPROT[1]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_axi_interface has unconnected port S_AXI_ARPROT[0]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_lcm has unconnected port s00_bram_addr[1]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_lcm has unconnected port s00_bram_addr[0]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_lcm has unconnected port s00_bram_rst}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_delorean_lcm_0_0 has unconnected port s00_axi_awprot[2]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_delorean_lcm_0_0 has unconnected port s00_axi_awprot[1]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_delorean_lcm_0_0 has unconnected port s00_axi_awprot[0]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_delorean_lcm_0_0 has unconnected port s00_axi_arprot[2]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_delorean_lcm_0_0 has unconnected port s00_axi_arprot[1]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_delorean_lcm_0_0 has unconnected port s00_axi_arprot[0]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_delorean_lcm_0_0 has unconnected port s00_bram_addr[1]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_delorean_lcm_0_0 has unconnected port s00_bram_addr[0]}}
set_msg_config -new_severity INFO -id {Synth 8-3331} -string {{WARNING: [Synth 8-3331] design delorean_delorean_lcm_0_0 has unconnected port s00_bram_rst}}

# Issues when packaging the ip project.
set_msg_config -new_severity INFO -id {IP_Flow 19-3153} -string {{WARNING: [IP_Flow 19-3153] Bus Interface 'ITO_CLK': ASSOCIATED_BUSIF bus parameter is missing.}}
set_msg_config -new_severity INFO -id {IP_Flow 19-4751} -string {{WARNING: [IP_Flow 19-4751] Bus Interface 'ITO_CLK': FREQ_HZ bus parameter is missing for output clock interface.}}
set_msg_config -new_severity INFO -id {IP_Flow 19-3153} -string {{WARNING: [IP_Flow 19-3153] Bus Interface 'RX_LVDS_CLK_N': ASSOCIATED_BUSIF bus parameter is missing.}}
set_msg_config -new_severity INFO -id {IP_Flow 19-4751} -string {{WARNING: [IP_Flow 19-4751] Bus Interface 'RX_LVDS_CLK_N': FREQ_HZ bus parameter is missing for output clock interface.}}
set_msg_config -new_severity INFO -id {IP_Flow 19-3153} -string {{WARNING: [IP_Flow 19-3153] Bus Interface 'RX_LVDS_CLK_P': ASSOCIATED_BUSIF bus parameter is missing.}}
set_msg_config -new_severity INFO -id {IP_Flow 19-4751} -string {{WARNING: [IP_Flow 19-4751] Bus Interface 'RX_LVDS_CLK_P': FREQ_HZ bus parameter is missing for output clock interface.}}
set_msg_config -new_severity INFO -id {IP_Flow 19-3153} -string {{WARNING: [IP_Flow 19-3153] Bus Interface 'TX_LVDS_CLK_N': ASSOCIATED_BUSIF bus parameter is missing.}}
set_msg_config -new_severity INFO -id {IP_Flow 19-4751} -string {{WARNING: [IP_Flow 19-4751] Bus Interface 'TX_LVDS_CLK_N': FREQ_HZ bus parameter is missing for output clock interface.}}
set_msg_config -new_severity INFO -id {IP_Flow 19-3153} -string {{WARNING: [IP_Flow 19-3153] Bus Interface 'TX_LVDS_CLK_P': ASSOCIATED_BUSIF bus parameter is missing.}}
set_msg_config -new_severity INFO -id {IP_Flow 19-4751} -string {{WARNING: [IP_Flow 19-4751] Bus Interface 'TX_LVDS_CLK_P': FREQ_HZ bus parameter is missing for output clock interface.}}

# single outputs come out of rX side IOB
set_msg_config -new_severity INFO -id {Opt 31-33} -string {{WARNING: [Opt 31-33] Removing redundant OBUF since it is not driving a top-level port. U_delorean_lcm_io_mlvds_tx/U_OBUF_LCD_EN}}
set_msg_config -new_severity INFO -id {Constraints 18-549} -string {{WARNING: [Constraints 18-549] Could not create 'DRIVE' constraint because cell 'U_delorean_lcm_io_mlvds_tx/U_OBUFT_ITO_CLK' is not directly connected to top level port. 'DRIVE' is ignored by Vivado but preserved inside the database.}}
set_msg_config -new_severity INFO -id {Constraints 18-549} -string {{WARNING: [Constraints 18-549] Could not create 'IOSTANDARD' constraint because cell 'U_delorean_lcm_io_mlvds_tx/U_OBUFT_ITO_CLK' is not directly connected to top level port. 'IOSTANDARD' is ignored by Vivado but preserved inside the database.}}
set_msg_config -new_severity INFO -id {Constraints 18-549} -string {{WARNING: [Constraints 18-549] Could not create 'SLEW' constraint because cell 'U_delorean_lcm_io_mlvds_tx/U_OBUFT_ITO_CLK' is not directly connected to top level port. 'SLEW' is ignored by Vivado but preserved inside the database.}}
set_msg_config -new_severity INFO -id {Constraints 18-549} -string {{WARNING: [Constraints 18-549] Could not create 'DRIVE' constraint because cell 'U_delorean_lcm_io_mlvds_tx/U_OBUFT_PROG_TRIGGER' is not directly connected to top level port. 'DRIVE' is ignored by Vivado but preserved inside the database.}}
set_msg_config -new_severity INFO -id {Constraints 18-549} -string {{WARNING: [Constraints 18-549] Could not create 'IOSTANDARD' constraint because cell 'U_delorean_lcm_io_mlvds_tx/U_OBUFT_PROG_TRIGGER' is not directly connected to top level port. 'IOSTANDARD' is ignored by Vivado but preserved inside the database.}}
set_msg_config -new_severity INFO -id {Constraints 18-549} -string {{WARNING: [Constraints 18-549] Could not create 'SLEW' constraint because cell 'U_delorean_lcm_io_mlvds_tx/U_OBUFT_PROG_TRIGGER' is not directly connected to top level port. 'SLEW' is ignored by Vivado but preserved inside the database.}}
