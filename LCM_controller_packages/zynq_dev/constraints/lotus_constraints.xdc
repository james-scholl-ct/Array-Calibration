# -----------------------------------------------------------------------------
# Pin constraints
# -----------------------------------------------------------------------------
# Mini-LVDS
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_CLK_P]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_CLK_N]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_0P]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_0N]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_1P]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_1N]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_2P]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_2N]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_3P]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_3N]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_4P]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_4N]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_5P]
set_property IOSTANDARD MINI_LVDS_25 [get_ports LVDS_5N]
set_property IOSTANDARD LVCMOS33 [get_ports POL]
set_property IOSTANDARD LVCMOS33 [get_ports TP1]
# BSC GPO
set_property IOSTANDARD LVCMOS33 [get_ports STATE_EQ_PROG]
set_property IOSTANDARD LVCMOS33 [get_ports ITO_CLK]
# Laser
set_property IOSTANDARD LVDS_25 [get_ports LASER_DR1_P]
set_property IOSTANDARD LVDS_25 [get_ports LASER_DR1_N]
set_property IOSTANDARD LVDS_25 [get_ports LASER_DR2_P]
set_property IOSTANDARD LVDS_25 [get_ports LASER_DR2_N]
set_property IOSTANDARD LVCMOS33 [get_ports LASER_TRIGGER]
# SPI
set_property IOSTANDARD LVCMOS33 [get_ports SPI_ADC0_CHSEL]
set_property IOSTANDARD LVCMOS33 [get_ports SPI_DAISY_CLK_EN]
set_property IOSTANDARD LVCMOS33 [get_ports SPI_DAISY_EN]
set_property IOSTANDARD LVCMOS33 [get_ports SPI_MISO]
set_property IOSTANDARD LVCMOS33 [get_ports SPI_MOSI]
set_property IOSTANDARD LVCMOS33 [get_ports SPI_SCLK0]
set_property IOSTANDARD LVCMOS33 [get_ports SPI_SCLK1]
set_property IOSTANDARD LVCMOS33 [get_ports SPI_SS_B]
# ONEWIRE
set_property IOSTANDARD LVCMOS33 [get_ports ONEWIRE_TEMP]
set_property IOSTANDARD LVCMOS33 [get_ports ONEWIRE_PULLUP_EN_B]

# Mini-LVDS
set_property PACKAGE_PIN V16 [get_ports LVDS_CLK_P]
set_property PACKAGE_PIN W16 [get_ports LVDS_CLK_N]
set_property PACKAGE_PIN U18 [get_ports LVDS_0P]
set_property PACKAGE_PIN U19 [get_ports LVDS_0N]
set_property PACKAGE_PIN N20 [get_ports LVDS_1P]
set_property PACKAGE_PIN P20 [get_ports LVDS_1N]
set_property PACKAGE_PIN V20 [get_ports LVDS_2P]
set_property PACKAGE_PIN W20 [get_ports LVDS_2N]
set_property PACKAGE_PIN T17 [get_ports LVDS_3P]
set_property PACKAGE_PIN R18 [get_ports LVDS_3N]
set_property PACKAGE_PIN W18 [get_ports LVDS_4P]
set_property PACKAGE_PIN W19 [get_ports LVDS_4N]
set_property PACKAGE_PIN P15 [get_ports LVDS_5P]
set_property PACKAGE_PIN P16 [get_ports LVDS_5N]
set_property PACKAGE_PIN G18 [get_ports POL]
set_property PACKAGE_PIN G19 [get_ports TP1]
# BSC GPO
set_property PACKAGE_PIN M18 [get_ports STATE_EQ_PROG]
set_property PACKAGE_PIN L19 [get_ports ITO_CLK]
# Laser
set_property PACKAGE_PIN T12 [get_ports LASER_DR1_P]
set_property PACKAGE_PIN U12 [get_ports LASER_DR1_N]
set_property PACKAGE_PIN T11 [get_ports LASER_DR2_P]
set_property PACKAGE_PIN T10 [get_ports LASER_DR2_N]
set_property PACKAGE_PIN A20 [get_ports LASER_TRIGGER]
# SPI
set_property PACKAGE_PIN E18 [get_ports SPI_ADC0_CHSEL]
set_property PACKAGE_PIN G20 [get_ports SPI_DAISY_CLK_EN]
set_property PACKAGE_PIN M14 [get_ports SPI_DAISY_EN]
set_property PACKAGE_PIN B20 [get_ports {SPI_MISO[0]}]
set_property PACKAGE_PIN H15 [get_ports {SPI_MISO[1]}]
set_property PACKAGE_PIN L15 [get_ports {SPI_MOSI[0]}]
set_property PACKAGE_PIN K16 [get_ports {SPI_MOSI[1]}]
set_property PACKAGE_PIN F17 [get_ports SPI_SCLK0]
set_property PACKAGE_PIN J16 [get_ports SPI_SCLK1]
set_property PACKAGE_PIN E17 [get_ports {SPI_SS_B[0]}]
set_property PACKAGE_PIN D18 [get_ports {SPI_SS_B[1]}]
set_property PACKAGE_PIN N16 [get_ports {SPI_SS_B[2]}]
set_property PACKAGE_PIN L14 [get_ports {SPI_SS_B[3]}]
set_property PACKAGE_PIN J14 [get_ports {SPI_SS_B[4]}]
# ONEWIRE
set_property PACKAGE_PIN D19 [get_ports {ONEWIRE_TEMP[0]}]
set_property PACKAGE_PIN D20 [get_ports {ONEWIRE_TEMP[1]}]
set_property PACKAGE_PIN F16 [get_ports {ONEWIRE_TEMP[2]}]
set_property PACKAGE_PIN M19 [get_ports ONEWIRE_PULLUP_EN_B]


# -----------------------------------------------------------------------------
# Unused Pins
# -----------------------------------------------------------------------------
# TODO
#
# Lotus          JX2 Pin  MicroZed Net     Zynq Pkg  Zynq Pin
# -------------  -------  ---------------  --------  ------------------------
# LED0           13       JX2_SE_0         G14       IO_0_35
# LED1           14       JX2_SE_1         J15       IO_25_35
# BUTTON0        17       JX2_LVDS_0_P     C20       IO_L1P_T0_AD0P_35
# BUTTON1        18       JX2_LVDS_1_P     B19       IO_L2P_T0_AD8P_35


# GPIO: BANK 34
#
# Lotus          JX1 Pin  MicroZed Net     Zynq Pkg  Zynq Pin
# -------------  -------  ---------------  --------  ------------------------
# GPIO25_9       9        JX1_SE_0         R19       IO_0_34
# GPIO25_10      10       JX1_SE_1         T19       IO_25_34
# GPIO25_17      17       JX1_LVDS_2_P     U13       IO_L3P_T0_DQS_PUDC_B_34
# GPIO25_18      18       JX1_LVDS_3_P     V12       IO_L4P_T0_34
# GPIO25_19      19       JX1_LVDS_2_N     V13       IO_L3N_T0_DQS_34
# GPIO25_20      20       JX1_LVDS_3_N     W13       IO_L4N_T0_34
# GPIO25_23      23       JX1_LVDS_4_P     T14       IO_L5P_T0_34
# GPIO25_24      24       JX1_LVDS_5_P     P14       IO_L6P_T0_34
# GPIO25_25      25       JX1_LVDS_4_N     T15       IO_L5N_T0_34
# GPIO25_26      26       JX1_LVDS_5_N     R14       IO_L6N_T0_VREF_34
# GPIO25_29      29       JX1_LVDS_6_P     Y16       IO_L7P_T1_34
# GPIO25_30      30       JX1_LVDS_7_P     W14       IO_L8P_T1_34
# GPIO25_31      31       JX1_LVDS_6_N     Y17       IO_L7N_T1_34
# GPIO25_32      32       JX1_LVDS_7_N     Y14       IO_L8N_T1_34
# GPIO25_35      35       JX1_LVDS_8_P     T16       IO_L9P_T1_DQS_34
# GPIO25_36      36       JX1_LVDS_9_P     V15       IO_L10P_T1_34
# GPIO25_37      37       JX1_LVDS_8_N     U17       IO_L9N_T1_DQS_34
# GPIO25_38      38       JX1_LVDS_9_N     W15       IO_L10N_T1_34
# GPIO25_41      41       JX1_LVDS_10_P    U14       IO_L11P_T1_SRCC_34
# GPIO25_43      43       JX1_LVDS_10_N    U15       IO_L11N_T1_SRCC_34
# GPIO25_47      47       JX1_LVDS_12_P    N18       IO_L13P_T2_MRCC_34
# GPIO25_49      49       JX1_LVDS_12_N    P19       IO_L13N_T2_MRCC_34
# GPIO25_53      53       JX1_LVDS_14_P    T20       IO_L15P_T2_DQS_34
# GPIO25_55      55       JX1_LVDS_14_N    U20       IO_L15N_T2_DQS_34
# GPIO25_61      61       JX1_LVDS_16_P    Y18       IO_L17P_T2_34
# GPIO25_63      63       JX1_LVDS_16_N    Y19       IO_L17N_T2_34
# GPIO25_67      67       JX1_LVDS_18_P    R16       IO_L19P_T3_34
# GPIO25_69      69       JX1_LVDS_18_N    R17       IO_L19N_T3_VREF_34
# GPIO25_73      73       JX1_LVDS_20_P    V17       IO_L21P_T3_DQS_34
# GPIO25_75      75       JX1_LVDS_20_N    V18       IO_L21N_T3_DQS_34
# GPIO25_81      81       JX1_LVDS_22_P    N17       IO_L23P_T3_34
# GPIO25_83      83       JX1_LVDS_22_N    P18       IO_L23N_T3_34


# GPIO: BANK 35
#
# Lotus          JX2 Pin  MicroZed Net     Zynq Pkg  Zynq Pin
# -------------  -------  ---------------  --------  ------------------------
# GPIO33_31      31       JX2_LVDS_4_N     E19       IO_L5N_T0_AD9N_35
# GPIO33_37      37       JX2_LVDS_6_N     L20       IO_L9N_T1_DQS_AD3N_35
# GPIO33_38      38       JX2_LVDS_7_N     M20       IO_L7N_T1_AD2N_35
# GPIO33_41      41       JX2_LVDS_8_P     M17       IO_L8P_T1_AD10P_35
# GPIO33_42      42       JX2_LVDS_9_P     K19       IO_L10P_T1_AD11P_35
# GPIO33_44      44       JX2_LVDS_9_N     J19       IO_L10N_T1_AD11N_35
# GPIO33_47      47       JX2_LVDS_10_P    L16       IO_L11P_T1_SRCC_35
# GPIO33_48      48       JX2_LVDS_11_P    K17       IO_L12P_T1_MRCC_35
# GPIO33_49      49       JX2_LVDS_10_N    L17       IO_L11N_T1_SRCC_35
# GPIO33_50      50       JX2_LVDS_11_N    K18       IO_L12N_T1_MRCC_35
# GPIO33_53      53       JX2_LVDS_12_P    H16       IO_L13P_T2_MRCC_35
# GPIO33_54      54       JX2_LVDS_13_P    J18       IO_L14P_T2_AD4P_SRCC_35
# GPIO33_55      55       JX2_LVDS_12_N    H17       IO_L13N_T2_MRCC_35
# GPIO33_56      56       JX2_LVDS_13_N    H18       IO_L14N_T2_AD4N_SRCC_35
# GPIO33_61      61       JX2_LVDS_14_P    G17       IO_L16P_T2_35
# GPIO33_62      62       JX2_LVDS_15_P    F19       IO_L15P_T2_DQS_AD12P_35
# GPIO33_64      64       JX2_LVDS_15_N    F20       IO_L15N_T2_DQS_AD12N_35
# GPIO33_68      68       JX2_LVDS_17_P    J20       IO_L17P_T2_AD5P_35
# GPIO33_70      70       JX2_LVDS_17_N    H20       IO_L17N_T2_AD5N_35
# GPIO33_73      73       JX2_LVDS_18_P    K14       IO_L20P_T3_AD6P_35
# GPIO33_76      76       JX2_LVDS_19_N    G15       IO_L19N_T3_VREF_35
# GPIO33_81      81       JX2_LVDS_20_P    N15       IO_L21P_T3_DQS_AD14P_35
# GPIO33_89      89       JX2_LVDS_22_N    M15       IO_L23N_T3_35


## -----------------------------------------------------------------------------
## Timing constraints
## -----------------------------------------------------------------------------
# Config RAM is considered/required to be static at the time of access.
set_false_path \
    -from [get_pins {lotus_i/lotus_bsc_0/inst/I_lotus_active_mem/regfile_reg[61][*]/C}] \
    -to   [get_pins lotus_i/lotus_bsc_0/inst/I_lotus_core/I_lotus_fsm_scan_seq/dwell_expired_reg/D]

set_false_path \
    -from [get_pins {lotus_i/lotus_bsc_0/inst/I_lotus_active_mem/regfile_reg[61][*]/C}] \
    -to   [get_pins {lotus_i/lotus_bsc_0/inst/I_lotus_core/I_lotus_fsm_scan_seq/ito_cnt_reg[*]/R}]

set_false_path \
    -from [get_pins {lotus_i/lotus_bsc_0/inst/I_lotus_active_mem/regfile_reg[61][*]/C}] \
    -to   [get_pins {lotus_i/lotus_bsc_0/inst/I_lotus_core/I_lotus_fsm_scan_seq/ITO_CLK_reg/D}]

set_false_path \
    -from [get_pins {lotus_i/lotus_bsc_0/inst/I_lotus_active_mem/regfile_reg[61][*]/C}] \
    -to   [get_pins {lotus_i/lotus_bsc_0/inst/I_lotus_core/I_lotus_fsm_scan_seq/STATE_EQ_PROG_reg/D}]


# CDC handshake
set_false_path \
    -from [get_pins {lotus_i/lotus_bsc_0/inst/I_lotus_core/I_lotus_sync_a2b/ax_req_reg/C}] \
    -to   [get_pins {lotus_i/lotus_bsc_0/inst/I_lotus_core/I_lotus_sync_a2b/b_req_sync0_reg/D}]

set_false_path \
    -from [get_pins {lotus_i/lotus_bsc_0/inst/I_lotus_core/I_lotus_sync_a2b/b_req_sync2_reg/C}] \
    -to   [get_pins {lotus_i/lotus_bsc_0/inst/I_lotus_core/I_lotus_sync_a2b/a_ack_sync0_reg/D}]

# CDC data to transfer
set_false_path \
    -from [get_pins {lotus_i/lotus_bsc_0/inst/I_lotus_active_mem/regfile_reg[62][*]/C}] \
    -to   [get_pins {lotus_i/lotus_bsc_0/inst/I_lotus_core/I_lotus_sync_a2b/B_DATA_reg[*]/D}]

# CDC for status
set_false_path \
    -from [get_pins {lotus_i/lotus_bsc_0/inst/I_lotus_core/I_lotus_fsm_scan_seq/STATE_EQ_DONE_reg/C}] \
    -to   [get_pins {lotus_i/lotus_bsc_0/inst/I_lotus_core/state_eq_done_sync0_reg/D}]

# regexp matches all combinations of regfile_reg[0..50][0..31]
set_false_path \
    -from [get_pins -regexp {lotus_i/lotus_bsc_0/inst/I_lotus_active_mem/regfile_reg\[([0-9]|[1-4][0-9]|50)\]\[[0-9]+\]/C}] \
    -to   [get_pins {lotus_i/lotus_bsc_0/inst/I_lotus_core/I_lotus_sync_a2b/B_DATA2_reg[*]/D}]



# LVDS clock (for set_output_delay)
create_generated_clock \
    -name LVDS_CLK \
    -source [get_pins lotus_i/clk_wiz_0/clk_out1] \
    -edges {1 3 5} \
    -edge_shift {8.333 8.333 8.333} \
    [get_pins lotus_i/LVDS_CLK_P]

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
set_property ASYNC_REG TRUE [get_cells {lotus_i/lotus_bsc_0/inst/I_lotus_core/I_lotus_sync_a2b/b_req_sync0_reg}]
set_property ASYNC_REG TRUE [get_cells {lotus_i/lotus_bsc_0/inst/I_lotus_core/I_lotus_sync_a2b/b_req_sync1_reg}]
set_property ASYNC_REG TRUE [get_cells {lotus_i/lotus_bsc_0/inst/I_lotus_core/I_lotus_sync_a2b/b_req_sync2_reg}]
set_property ASYNC_REG TRUE [get_cells {lotus_i/lotus_bsc_0/inst/I_lotus_core/I_lotus_sync_a2b/a_ack_sync0_reg}]
set_property ASYNC_REG TRUE [get_cells {lotus_i/lotus_bsc_0/inst/I_lotus_core/I_lotus_sync_a2b/a_ack_sync1_reg}]
set_property ASYNC_REG TRUE [get_cells {lotus_i/lotus_bsc_0/inst/I_lotus_core/state_eq_done_sync0_reg}]
set_property ASYNC_REG TRUE [get_cells {lotus_i/lotus_bsc_0/inst/I_lotus_core/state_eq_done_sync1_reg}]
