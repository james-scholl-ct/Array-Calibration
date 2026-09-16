# -----------------------------------------------------------------------------
# Pin constraints
# -----------------------------------------------------------------------------
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_CLK_P]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_CLK_N]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_0N]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_0P]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_1N]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_1P]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_2N]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_2P]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_3N]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_3P]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_4N]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_4P]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_5N]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_5P]
set_property IOSTANDARD LVCMOS33 [get_ports POL]
set_property IOSTANDARD LVCMOS33 [get_ports TP1]

set_property PACKAGE_PIN Y16 [get_ports LVDS_CLK_P]
set_property PACKAGE_PIN Y17 [get_ports LVDS_CLK_N]
set_property PACKAGE_PIN T11 [get_ports LVDS_0P]
set_property PACKAGE_PIN T10 [get_ports LVDS_0N]
set_property PACKAGE_PIN T12 [get_ports LVDS_1P]
set_property PACKAGE_PIN U12 [get_ports LVDS_1N]
set_property PACKAGE_PIN U13 [get_ports LVDS_2P]
set_property PACKAGE_PIN V13 [get_ports LVDS_2N]
set_property PACKAGE_PIN V12 [get_ports LVDS_3P]
set_property PACKAGE_PIN W13 [get_ports LVDS_3N]
set_property PACKAGE_PIN T14 [get_ports LVDS_4P]
set_property PACKAGE_PIN T15 [get_ports LVDS_4N]
set_property PACKAGE_PIN P14 [get_ports LVDS_5P]
set_property PACKAGE_PIN R14 [get_ports LVDS_5N]
set_property PACKAGE_PIN G14 [get_ports POL]
set_property PACKAGE_PIN J15 [get_ports TP1]


# -----------------------------------------------------------------------------
# Timing constraints
# -----------------------------------------------------------------------------
# Coefficient RAM is considered/required to be static at the time of access.
#   regexp matches all combinations of slv_regs[0..63][0..31]
set_false_path \
    -from [get_pins -regexp {orchid_i/orchid_0/inst/I_orchid_axi_interface/slv_regs_reg\[([0-9]|[1-5][0-9]|6[0-3])\]\[[0-9]+\]/C}] \
    -to   [get_pins {orchid_i/orchid_0/inst/I_orchid_core/I_orchid_data_mux/LVDS_DATA_reg[*]/D}]

# Config RAM is considered/required to be static at the time of access.
#   regexp matches all combinations of slv_regs[68..69][0..31]
set_false_path \
    -from [get_pins -regexp {orchid_i/orchid_0/inst/I_orchid_axi_interface/slv_regs_reg\[6[89]\]\[[0-9]+\]/C}] \
    -to   [get_pins orchid_i/orchid_0/inst/I_orchid_core/I_orchid_fsm_scan_seq/dwell_expired_reg/D]

# CDC handshake
set_false_path \
    -from [get_pins {orchid_i/orchid_0/inst/I_orchid_sync_a2b/a2b_req_reg/C}] \
    -to   [get_pins {orchid_i/orchid_0/inst/I_orchid_sync_a2b/a2b_req_sync0_reg/D}]

set_false_path \
    -from [get_pins {orchid_i/orchid_0/inst/I_orchid_sync_a2b/a2b_req_sync1_reg/C}] \
    -to   [get_pins {orchid_i/orchid_0/inst/I_orchid_sync_a2b/b2a_ack_sync0_reg/D}]

# CDC data to transfer
set_false_path \
    -from [get_pins {orchid_i/orchid_0/inst/I_orchid_axi_interface/slv_regs_reg[71][*]/C}] \
    -to   [get_pins {orchid_i/orchid_0/inst/I_orchid_sync_a2b/B_DATA_reg[*]/D}]

# TABLE_SEL is constant during programming of lvds data (it only transitions
# in the WAIT state and with global reset). Use MCP instead of false path to
# be safe.
set_multicycle_path 2 -setup \
    -from [get_pins {orchid_i/orchid_0/inst/I_orchid_core/I_orchid_fsm_scan_seq/TABLE_SEL_reg[*]*/C}] \
    -to   [get_pins {orchid_i/orchid_0/inst/I_orchid_core/I_orchid_data_mux/LVDS_DATA_reg[*]/D}]

set_multicycle_path 1 -hold \
    -from [get_pins {orchid_i/orchid_0/inst/I_orchid_core/I_orchid_fsm_scan_seq/TABLE_SEL_reg[*]*/C}] \
    -to   [get_pins {orchid_i/orchid_0/inst/I_orchid_core/I_orchid_data_mux/LVDS_DATA_reg[*]/D}]

# LVDS clock (for set_output_delay)
create_generated_clock \
    -name LVDS_CLK \
    -source [get_pins orchid_i/clk_wiz_0/clk_out1] \
    -edges {1 3 5} \
    -edge_shift {8.333 8.333 8.333} \
    [get_pins orchid_i/LVDS_CLK_P]

# Clock routed with data so don't worry about this
set_output_delay \
    -clock [get_clocks LVDS_CLK] \
    0.000 [get_ports -regexp -filter {NAME =~ "LVDS_[0-6][PN]" && DIRECTION == "OUT"}]

set_output_delay \
    -clock [get_clocks LVDS_CLK] \
    0.000 [get_ports -filter {DIRECTION == "OUT"} {POL TP1}]


# -----------------------------------------------------------------------------
# Placement and Simulation constraints
# -----------------------------------------------------------------------------
# Place ASYNC_REG property on synchronizing flops (not synchronized flops)
set_property ASYNC_REG TRUE [get_cells {orchid_i/orchid_0/inst/I_orchid_sync_a2b/a2b_req_sync0_reg}]
set_property ASYNC_REG TRUE [get_cells {orchid_i/orchid_0/inst/I_orchid_sync_a2b/a2b_req_sync1_reg}]
set_property ASYNC_REG TRUE [get_cells {orchid_i/orchid_0/inst/I_orchid_sync_a2b/a2b_req_sync2_reg}]
set_property ASYNC_REG TRUE [get_cells {orchid_i/orchid_0/inst/I_orchid_sync_a2b/b2a_ack_sync0_reg}]
set_property ASYNC_REG TRUE [get_cells {orchid_i/orchid_0/inst/I_orchid_sync_a2b/b2a_ack_sync1_reg}]
