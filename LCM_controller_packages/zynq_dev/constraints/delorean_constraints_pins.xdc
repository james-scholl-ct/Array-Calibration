# =============================================================================
# Pin Constraints: Bank 34 (VCCO = 2.5V)
# =============================================================================
# -----------------------------------------------------------------------------
# Map the Microzed JX1 connector pins to I/O port names in the design. The even
# pin numbers are on one side of the connector and the odds are on the other.
# -----------------------------------------------------------------------------
set JX1_09_SE_0         {GPIO_0}
set JX1_10_SE_1         {GPIO_11}

set JX1_11_LVDS_0_P     {GPIO_1}
set JX1_13_LVDS_0_N     {GPIO_2}
set JX1_12_LVDS_1_P     {GPIO_12}
set JX1_14_LVDS_1_N     {GPIO_13}

set JX1_17_LVDS_2_P     {GPIO_3}
set JX1_19_LVDS_2_N     {GPIO_4}
set JX1_18_LVDS_3_P     {GPIO_7}
set JX1_20_LVDS_3_N     {GPIO_8}

set JX1_23_LVDS_4_P     {TX_LVDS_DATA_P[5]}
set JX1_25_LVDS_4_N     {TX_LVDS_DATA_N[5]}
set JX1_24_LVDS_5_P     {RX_LVDS_DATA_P[5]}
set JX1_26_LVDS_5_N     {RX_LVDS_DATA_N[5]}

set JX1_29_LVDS_6_P     {TX_LVDS_DATA_P[4]}
set JX1_31_LVDS_6_N     {TX_LVDS_DATA_N[4]}
set JX1_30_LVDS_7_P     {RX_LVDS_DATA_P[4]}
set JX1_32_LVDS_7_N     {RX_LVDS_DATA_N[4]}

set JX1_35_LVDS_8_P     {TX_LVDS_DATA_P[3]}
set JX1_37_LVDS_8_N     {TX_LVDS_DATA_N[3]}
set JX1_36_LVDS_9_P     {RX_LVDS_DATA_P[3]}
set JX1_38_LVDS_9_N     {RX_LVDS_DATA_N[3]}

set JX1_41_LVDS_10_P    {TX_LVDS_DATA_P[2]}
set JX1_43_LVDS_10_N    {TX_LVDS_DATA_N[2]}
set JX1_42_LVDS_11_P    {RX_LVDS_CLK_P}
set JX1_44_LVDS_11_N    {RX_LVDS_CLK_N}

set JX1_47_LVDS_12_P    {TX_LVDS_CLK_P}
set JX1_49_LVDS_12_N    {TX_LVDS_CLK_N}
set JX1_48_LVDS_13_P    {RX_LVDS_DATA_P[2]}
set JX1_50_LVDS_13_N    {RX_LVDS_DATA_N[2]}

set JX1_53_LVDS_14_P    {TX_LVDS_DATA_P[1]}
set JX1_55_LVDS_14_N    {TX_LVDS_DATA_N[1]}
set JX1_54_LVDS_15_P    {RX_LVDS_DATA_P[1]}
set JX1_56_LVDS_15_N    {RX_LVDS_DATA_N[1]}

set JX1_61_LVDS_16_P    {TX_LVDS_DATA_P[0]}
set JX1_63_LVDS_16_N    {TX_LVDS_DATA_N[0]}
set JX1_62_LVDS_17_P    {RX_LVDS_DATA_P[0]}
set JX1_64_LVDS_17_N    {RX_LVDS_DATA_N[0]}

set JX1_67_LVDS_18_P    {GPIO_5}
set JX1_69_LVDS_18_N    {GPIO_6}
set JX1_68_LVDS_19_P    {LASER_DR1_P[0]}
set JX1_70_LVDS_19_N    {LASER_DR1_N[0]}

set JX1_73_LVDS_20_P    {GPIO_9}
set JX1_75_LVDS_20_N    {GPIO_10}
set JX1_74_LVDS_21_P    {LASER_DR2_P[0]}
set JX1_76_LVDS_21_N    {LASER_DR2_N[0]}

set JX1_81_LVDS_22_P    {LASER_DR2_P[1]}
set JX1_83_LVDS_22_N    {LASER_DR2_N[1]}
set JX1_82_LVDS_23_P    {LASER_DR1_P[1]}
set JX1_84_LVDS_23_N    {LASER_DR1_N[1]}

# -----------------------------------------------------------------------------
# Uncomment each line per the allocation above
# -----------------------------------------------------------------------------
#set_property PACKAGE_PIN R19 [get_ports $JX1_09_SE_0]
#set_property PACKAGE_PIN T19 [get_ports $JX1_10_SE_1]

#set_property PACKAGE_PIN T11 [get_ports $JX1_11_LVDS_0_P]
#set_property PACKAGE_PIN T10 [get_ports $JX1_13_LVDS_0_N]
#set_property PACKAGE_PIN T12 [get_ports $JX1_12_LVDS_1_P]
#set_property PACKAGE_PIN U12 [get_ports $JX1_14_LVDS_1_N]

#set_property PACKAGE_PIN U13 [get_ports $JX1_17_LVDS_2_P]
#set_property PACKAGE_PIN V13 [get_ports $JX1_19_LVDS_2_N]
#set_property PACKAGE_PIN V12 [get_ports $JX1_18_LVDS_3_P]
#set_property PACKAGE_PIN W13 [get_ports $JX1_20_LVDS_3_N]

set_property PACKAGE_PIN T14 [get_ports $JX1_23_LVDS_4_P]
set_property PACKAGE_PIN T15 [get_ports $JX1_25_LVDS_4_N]
set_property PACKAGE_PIN P14 [get_ports $JX1_24_LVDS_5_P]
set_property PACKAGE_PIN R14 [get_ports $JX1_26_LVDS_5_N]

set_property PACKAGE_PIN Y16 [get_ports $JX1_29_LVDS_6_P]
set_property PACKAGE_PIN Y17 [get_ports $JX1_31_LVDS_6_N]
set_property PACKAGE_PIN W14 [get_ports $JX1_30_LVDS_7_P]
set_property PACKAGE_PIN Y14 [get_ports $JX1_32_LVDS_7_N]

set_property PACKAGE_PIN T16 [get_ports $JX1_35_LVDS_8_P]
set_property PACKAGE_PIN U17 [get_ports $JX1_37_LVDS_8_N]
set_property PACKAGE_PIN V15 [get_ports $JX1_36_LVDS_9_P]
set_property PACKAGE_PIN W15 [get_ports $JX1_38_LVDS_9_N]

set_property PACKAGE_PIN U14 [get_ports $JX1_41_LVDS_10_P]
set_property PACKAGE_PIN U15 [get_ports $JX1_43_LVDS_10_N]
set_property PACKAGE_PIN U18 [get_ports $JX1_42_LVDS_11_P]
set_property PACKAGE_PIN U19 [get_ports $JX1_44_LVDS_11_N]

set_property PACKAGE_PIN N18 [get_ports $JX1_47_LVDS_12_P]
set_property PACKAGE_PIN P19 [get_ports $JX1_49_LVDS_12_N]
set_property PACKAGE_PIN N20 [get_ports $JX1_48_LVDS_13_P]
set_property PACKAGE_PIN P20 [get_ports $JX1_50_LVDS_13_N]

set_property PACKAGE_PIN T20 [get_ports $JX1_53_LVDS_14_P]
set_property PACKAGE_PIN U20 [get_ports $JX1_55_LVDS_14_N]
set_property PACKAGE_PIN V20 [get_ports $JX1_54_LVDS_15_P]
set_property PACKAGE_PIN W20 [get_ports $JX1_56_LVDS_15_N]

set_property PACKAGE_PIN Y18 [get_ports $JX1_61_LVDS_16_P]
set_property PACKAGE_PIN Y19 [get_ports $JX1_63_LVDS_16_N]
set_property PACKAGE_PIN V16 [get_ports $JX1_62_LVDS_17_P]
set_property PACKAGE_PIN W16 [get_ports $JX1_64_LVDS_17_N]

#set_property PACKAGE_PIN R16 [get_ports $JX1_67_LVDS_18_P]
#set_property PACKAGE_PIN R17 [get_ports $JX1_69_LVDS_18_N]
set_property PACKAGE_PIN T17 [get_ports $JX1_68_LVDS_19_P]
set_property PACKAGE_PIN R18 [get_ports $JX1_70_LVDS_19_N]

#set_property PACKAGE_PIN V17 [get_ports $JX1_73_LVDS_20_P]
#set_property PACKAGE_PIN V18 [get_ports $JX1_75_LVDS_20_N]
set_property PACKAGE_PIN W18 [get_ports $JX1_74_LVDS_21_P]
set_property PACKAGE_PIN W19 [get_ports $JX1_76_LVDS_21_N]

set_property PACKAGE_PIN N17 [get_ports $JX1_81_LVDS_22_P]
set_property PACKAGE_PIN P18 [get_ports $JX1_83_LVDS_22_N]
set_property PACKAGE_PIN P15 [get_ports $JX1_82_LVDS_23_P]
set_property PACKAGE_PIN P16 [get_ports $JX1_84_LVDS_23_N]

# -----------------------------------------------------------------------------
# Set IOSTANDARD
# -----------------------------------------------------------------------------
set_property IOSTANDARD LVDS_25 [get_ports {LASER_DR1_P[*]}]
set_property IOSTANDARD LVDS_25 [get_ports {LASER_DR1_N[*]}]
set_property IOSTANDARD LVDS_25 [get_ports {LASER_DR2_P[*]}]
set_property IOSTANDARD LVDS_25 [get_ports {LASER_DR2_N[*]}]
set_property IOSTANDARD MINI_LVDS_25 [get_ports {RX_LVDS_DATA_P[*]}]
set_property IOSTANDARD MINI_LVDS_25 [get_ports {RX_LVDS_DATA_N[*]}]
set_property IOSTANDARD MINI_LVDS_25 [get_ports {TX_LVDS_DATA_P[*]}]
set_property IOSTANDARD MINI_LVDS_25 [get_ports {TX_LVDS_DATA_N[*]}]
set_property IOSTANDARD MINI_LVDS_25 [get_ports RX_LVDS_CLK_P]
set_property IOSTANDARD MINI_LVDS_25 [get_ports RX_LVDS_CLK_N]
set_property IOSTANDARD MINI_LVDS_25 [get_ports TX_LVDS_CLK_P]
set_property IOSTANDARD MINI_LVDS_25 [get_ports TX_LVDS_CLK_N]
#set_property IOSTANDARD LVCMOS25 [get_ports GPIO_0]
#set_property IOSTANDARD LVCMOS25 [get_ports GPIO_11]
#set_property IOSTANDARD LVCMOS25 [get_ports GPIO_1]
#set_property IOSTANDARD LVCMOS25 [get_ports GPIO_2]
#set_property IOSTANDARD LVCMOS25 [get_ports GPIO_12]
#set_property IOSTANDARD LVCMOS25 [get_ports GPIO_13]
#set_property IOSTANDARD LVCMOS25 [get_ports GPIO_3]
#set_property IOSTANDARD LVCMOS25 [get_ports GPIO_4]
#set_property IOSTANDARD LVCMOS25 [get_ports GPIO_7]
#set_property IOSTANDARD LVCMOS25 [get_ports GPIO_8]
#set_property IOSTANDARD LVCMOS25 [get_ports GPIO_5]
#set_property IOSTANDARD LVCMOS25 [get_ports GPIO_6]
#set_property IOSTANDARD LVCMOS25 [get_ports GPIO_9]
#set_property IOSTANDARD LVCMOS25 [get_ports GPIO_10]

# =============================================================================
# Pin Constraints: Bank 35 (VCCO = 3.3V)
# =============================================================================
# -----------------------------------------------------------------------------
# Map the Microzed JX2 connector pins to I/O port names in the design. The even
# pin numbers are on one side of the connector and the odds are on the other.
# -----------------------------------------------------------------------------
set JX2_13_SE_0         {LED[0]}
set JX2_14_SE_1         {LED[1]}

set JX2_17_LVDS_0_P     {BUTTON[0]}
set JX2_19_LVDS_0_N     {SPI_MISO[0]}
set JX2_18_LVDS_1_P     {BUTTON[1]}
set JX2_20_LVDS_1_N     {LASER_TRIGGER}

set JX2_23_LVDS_2_P     {SPI_CS_B[0]}
set JX2_25_LVDS_2_N     {SPI_CS_B[1]}
set JX2_24_LVDS_3_P     {GPIO_43}
set JX2_26_LVDS_3_N     {GPIO_44}

set JX2_29_LVDS_4_P     {ITO_CLK}
set JX2_31_LVDS_4_N     {SPI_MISO[1]}
set JX2_30_LVDS_5_P     {GPIO_41}
set JX2_32_LVDS_5_N     {SPI_CS_B[4]}

set JX2_35_LVDS_6_P     {SPI_ADC_CH_SEL[0]}
set JX2_37_LVDS_6_N     {GPIO_26}
set JX2_36_LVDS_7_P     {RX_EIO2}
set JX2_38_LVDS_7_N     {RX_TP2}

set JX2_41_LVDS_8_P     {SPI_ADC_CH_SEL[1]}
set JX2_43_LVDS_8_N     {GPIO_25}
set JX2_42_LVDS_9_P     {TX_EIO2}
set JX2_44_LVDS_9_N     {TX_TP2}

set JX2_47_LVDS_10_P    {LCD_EN}
set JX2_49_LVDS_10_N    {GPIO_24}
set JX2_48_LVDS_11_P    {TX_PWR_SWITCH}
set JX2_50_LVDS_11_N    {GPIO_42}

set JX2_53_LVDS_12_P    {GPIO_23}
set JX2_55_LVDS_12_N    {GPIO_22}
set JX2_54_LVDS_13_P    {TX_PWR_EN}
set JX2_56_LVDS_13_N    {GPIO_39}

set JX2_61_LVDS_14_P    {GPIO_27}
set JX2_63_LVDS_14_N    {TX_POL}
set JX2_62_LVDS_15_P    {GPIO_40}
set JX2_64_LVDS_15_N    {RX_POL}

set JX2_67_LVDS_16_P    {TX_TP1}
set JX2_69_LVDS_16_N    {PROG_TRIGGER}
set JX2_68_LVDS_17_P    {RX_TP1}
set JX2_70_LVDS_17_N    {GPIO_37}

set JX2_73_LVDS_18_P    {SPI_CS_B[5]}
set JX2_75_LVDS_18_N    {SPI_CS_B[2]}
set JX2_74_LVDS_19_P    {GPIO_38}
set JX2_76_LVDS_19_N    {GPIO_35}

set JX2_81_LVDS_20_P    {GPIO_28}
set JX2_83_LVDS_20_N    {SPI_CS_B[3]}
set JX2_82_LVDS_21_P    {GPIO_36}
set JX2_84_LVDS_21_N    {SPI_MOSI[0]}

set JX2_87_LVDS_22_P    {SPI_DAISY_EN}
set JX2_89_LVDS_22_N    {SPI_SCLK[0]}
set JX2_88_LVDS_23_P    {SPI_MOSI[1]}
set JX2_90_LVDS_23_N    {SPI_SCLK[1]}

# -----------------------------------------------------------------------------
# Uncomment each line per the allocation above
# -----------------------------------------------------------------------------
#set_property PACKAGE_PIN G14 [get_ports $JX2_13_SE_0]
#set_property PACKAGE_PIN J15 [get_ports $JX2_14_SE_1]

#set_property PACKAGE_PIN C20 [get_ports $JX2_17_LVDS_0_P]
set_property PACKAGE_PIN B20 [get_ports $JX2_19_LVDS_0_N]
#set_property PACKAGE_PIN B19 [get_ports $JX2_18_LVDS_1_P]
set_property PACKAGE_PIN A20 [get_ports $JX2_20_LVDS_1_N]

set_property PACKAGE_PIN E17 [get_ports $JX2_23_LVDS_2_P]
set_property PACKAGE_PIN D18 [get_ports $JX2_25_LVDS_2_N]
#set_property PACKAGE_PIN D19 [get_ports $JX2_24_LVDS_3_P]
#set_property PACKAGE_PIN D20 [get_ports $JX2_26_LVDS_3_N]

set_property PACKAGE_PIN E18 [get_ports $JX2_29_LVDS_4_P]
set_property PACKAGE_PIN E19 [get_ports $JX2_31_LVDS_4_N]
#set_property PACKAGE_PIN F16 [get_ports $JX2_30_LVDS_5_P]
set_property PACKAGE_PIN F17 [get_ports $JX2_32_LVDS_5_N]

set_property PACKAGE_PIN L19 [get_ports $JX2_35_LVDS_6_P]
#set_property PACKAGE_PIN L20 [get_ports $JX2_37_LVDS_6_N]
#set_property PACKAGE_PIN M19 [get_ports $JX2_36_LVDS_7_P]
#set_property PACKAGE_PIN M20 [get_ports $JX2_38_LVDS_7_N]

set_property PACKAGE_PIN M17 [get_ports $JX2_41_LVDS_8_P]
#set_property PACKAGE_PIN M18 [get_ports $JX2_43_LVDS_8_N]
#set_property PACKAGE_PIN K19 [get_ports $JX2_42_LVDS_9_P]
#set_property PACKAGE_PIN J19 [get_ports $JX2_44_LVDS_9_N]

set_property PACKAGE_PIN L16 [get_ports $JX2_47_LVDS_10_P]
#set_property PACKAGE_PIN L17 [get_ports $JX2_49_LVDS_10_N]
set_property PACKAGE_PIN K17 [get_ports $JX2_48_LVDS_11_P]
#set_property PACKAGE_PIN K18 [get_ports $JX2_50_LVDS_11_N]

#set_property PACKAGE_PIN H16 [get_ports $JX2_53_LVDS_12_P]
#set_property PACKAGE_PIN H17 [get_ports $JX2_55_LVDS_12_N]
set_property PACKAGE_PIN J18 [get_ports $JX2_54_LVDS_13_P]
#set_property PACKAGE_PIN H18 [get_ports $JX2_56_LVDS_13_N]

#set_property PACKAGE_PIN G17 [get_ports $JX2_61_LVDS_14_P]
set_property PACKAGE_PIN G18 [get_ports $JX2_63_LVDS_14_N]
#et_property PACKAGE_PIN F19 [get_ports $JX2_62_LVDS_15_P]
set_property PACKAGE_PIN F20 [get_ports $JX2_64_LVDS_15_N]

set_property PACKAGE_PIN G19 [get_ports $JX2_67_LVDS_16_P]
set_property PACKAGE_PIN G20 [get_ports $JX2_69_LVDS_16_N]
set_property PACKAGE_PIN J20 [get_ports $JX2_68_LVDS_17_P]
#set_property PACKAGE_PIN H20 [get_ports $JX2_70_LVDS_17_N]

set_property PACKAGE_PIN K14 [get_ports $JX2_73_LVDS_18_P]
set_property PACKAGE_PIN J14 [get_ports $JX2_75_LVDS_18_N]
#set_property PACKAGE_PIN H15 [get_ports $JX2_74_LVDS_19_P]
#set_property PACKAGE_PIN G15 [get_ports $JX2_76_LVDS_19_N]

#set_property PACKAGE_PIN N15 [get_ports $JX2_81_LVDS_20_P]
set_property PACKAGE_PIN N16 [get_ports $JX2_83_LVDS_20_N]
#set_property PACKAGE_PIN L14 [get_ports $JX2_82_LVDS_21_P]
set_property PACKAGE_PIN L15 [get_ports $JX2_84_LVDS_21_N]

set_property PACKAGE_PIN M14 [get_ports $JX2_87_LVDS_22_P]
set_property PACKAGE_PIN M15 [get_ports $JX2_89_LVDS_22_N]
set_property PACKAGE_PIN K16 [get_ports $JX2_88_LVDS_23_P]
set_property PACKAGE_PIN J16 [get_ports $JX2_90_LVDS_23_N]

# -----------------------------------------------------------------------------
# Set IOSTANDARD
# -----------------------------------------------------------------------------
set_property IOSTANDARD LVCMOS33 [get_ports LCD_EN]
set_property IOSTANDARD LVCMOS33 [get_ports TX_PWR_EN]
set_property IOSTANDARD LVCMOS33 [get_ports RX_POL]
set_property IOSTANDARD LVCMOS33 [get_ports TX_POL]
set_property IOSTANDARD LVCMOS33 [get_ports RX_TP1]
set_property IOSTANDARD LVCMOS33 [get_ports TX_TP1]
set_property IOSTANDARD LVCMOS33 [get_ports ITO_CLK]
set_property IOSTANDARD LVCMOS33 [get_ports PROG_TRIGGER]
set_property IOSTANDARD LVCMOS33 [get_ports {SPI_SCLK[*]}]
set_property IOSTANDARD LVCMOS33 [get_ports {SPI_MISO[*]}]
set_property IOSTANDARD LVCMOS33 [get_ports {SPI_MOSI[*]}]
set_property IOSTANDARD LVCMOS33 [get_ports {SPI_CS_B[*]}]
set_property IOSTANDARD LVCMOS33 [get_ports {SPI_ADC_CH_SEL[*]}]
set_property IOSTANDARD LVCMOS33 [get_ports SPI_DAISY_EN]
set_property IOSTANDARD LVCMOS33 [get_ports TX_PWR_SWITCH]
set_property IOSTANDARD LVCMOS33 [get_ports LASER_TRIGGER]
#set_property IOSTANDARD LVCMOS33 [get_ports RX_EIO2]
#set_property IOSTANDARD LVCMOS33 [get_ports TX_EIO2]
#set_property IOSTANDARD LVCMOS33 [get_ports RX_TP2]
#set_property IOSTANDARD LVCMOS33 [get_ports TX_TP2]
#set_property IOSTANDARD LVCMOS33 [get_ports {BUTTON[*]}]
#set_property IOSTANDARD LVCMOS33 [get_ports {LED[*]}]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_43]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_44]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_41]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_26]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_25]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_24]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_42]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_23]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_22]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_39]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_27]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_40]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_37]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_38]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_35]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_28]
#set_property IOSTANDARD LVCMOS33 [get_ports GPIO_36]

# =============================================================================
# Pullup and pulldown for tri-state outputs (see UG912)
# =============================================================================
#set_property PULLTYPE PULLDOWN [get_ports LCD_EN]
#set_property PULLTYPE PULLDOWN [get_ports TX_PWR_EN]
