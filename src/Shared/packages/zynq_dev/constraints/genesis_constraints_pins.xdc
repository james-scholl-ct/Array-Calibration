# =============================================================================
# Pin Constraints: Bank 34
# =============================================================================
# Map the Microzed JX1 connector pins to I/O port names in the design
#   This portion is pinned out to headers on Lotus
set JX1_09_SE_0         {GPI_IP1[1]}
set JX1_10_SE_1         {GPI_IP1[0]}
set JX1_23_LVDS_4_P     {GPI_IP0[1]}
set JX1_24_LVDS_5_P     {GPI_IP0[0]}
set JX1_25_LVDS_4_N     {GPO_IP1[1]}
set JX1_26_LVDS_5_N     {GPO_IP1[0]}
set JX1_29_LVDS_6_P     {GPO_IP0[1]}
set JX1_30_LVDS_7_P     {GPO_IP0[0]}
set JX1_31_LVDS_6_N     {}
set JX1_32_LVDS_7_N     {}
set JX1_35_LVDS_8_P     {}
set JX1_36_LVDS_9_P     {}
set JX1_37_LVDS_8_N     {}
set JX1_38_LVDS_9_N     {}
set JX1_41_LVDS_10_P    {}
set JX1_43_LVDS_10_N    {}
set JX1_47_LVDS_12_P    {}
set JX1_49_LVDS_12_N    {}
set JX1_53_LVDS_14_P    {}
set JX1_55_LVDS_14_N    {}
set JX1_61_LVDS_16_P    {}
set JX1_63_LVDS_16_N    {}

#   This portion is NOT pinned out to headers on Lotus
set JX1_11_LVDS_0_P     {}
set JX1_12_LVDS_1_P     {}
set JX1_13_LVDS_0_N     {}
set JX1_14_LVDS_1_N     {}
set JX1_17_LVDS_2_P     {}
set JX1_18_LVDS_3_P     {}
set JX1_19_LVDS_2_N     {}
set JX1_20_LVDS_3_N     {}
set JX1_42_LVDS_11_P    {}
set JX1_44_LVDS_11_N    {}
set JX1_48_LVDS_13_P    {}
set JX1_50_LVDS_13_N    {}
set JX1_54_LVDS_15_P    {}
set JX1_56_LVDS_15_N    {}
set JX1_62_LVDS_17_P    {}
set JX1_64_LVDS_17_N    {}
set JX1_67_LVDS_18_P    {}
set JX1_68_LVDS_19_P    {}
set JX1_69_LVDS_18_N    {}
set JX1_70_LVDS_19_N    {}
set JX1_73_LVDS_20_P    {}
set JX1_74_LVDS_21_P    {}
set JX1_75_LVDS_20_N    {}
set JX1_76_LVDS_21_N    {}
set JX1_81_LVDS_22_P    {}
set JX1_82_LVDS_23_P    {}
set JX1_83_LVDS_22_N    {}
set JX1_84_LVDS_23_N    {}


# Uncomment each line per the allocation above
set_property PACKAGE_PIN R19 [get_ports $JX1_09_SE_0]
set_property PACKAGE_PIN T19 [get_ports $JX1_10_SE_1]
#set_property PACKAGE_PIN T11 [get_ports $JX1_11_LVDS_0_P]
#set_property PACKAGE_PIN T12 [get_ports $JX1_12_LVDS_1_P]
#set_property PACKAGE_PIN T10 [get_ports $JX1_13_LVDS_0_N]
#set_property PACKAGE_PIN U12 [get_ports $JX1_14_LVDS_1_N]
#set_property PACKAGE_PIN U13 [get_ports $JX1_17_LVDS_2_P]
#set_property PACKAGE_PIN V12 [get_ports $JX1_18_LVDS_3_P]
#set_property PACKAGE_PIN V13 [get_ports $JX1_19_LVDS_2_N]
#set_property PACKAGE_PIN W13 [get_ports $JX1_20_LVDS_3_N]
set_property PACKAGE_PIN T14 [get_ports $JX1_23_LVDS_4_P]
set_property PACKAGE_PIN P14 [get_ports $JX1_24_LVDS_5_P]
set_property PACKAGE_PIN T15 [get_ports $JX1_25_LVDS_4_N]
set_property PACKAGE_PIN R14 [get_ports $JX1_26_LVDS_5_N]
set_property PACKAGE_PIN Y16 [get_ports $JX1_29_LVDS_6_P]
set_property PACKAGE_PIN W14 [get_ports $JX1_30_LVDS_7_P]
#set_property PACKAGE_PIN Y17 [get_ports $JX1_31_LVDS_6_N]
#set_property PACKAGE_PIN Y14 [get_ports $JX1_32_LVDS_7_N]
#set_property PACKAGE_PIN T16 [get_ports $JX1_35_LVDS_8_P]
#set_property PACKAGE_PIN V15 [get_ports $JX1_36_LVDS_9_P]
#set_property PACKAGE_PIN U17 [get_ports $JX1_37_LVDS_8_N]
#set_property PACKAGE_PIN W15 [get_ports $JX1_38_LVDS_9_N]
#set_property PACKAGE_PIN U14 [get_ports $JX1_41_LVDS_10_P]
#set_property PACKAGE_PIN U18 [get_ports $JX1_42_LVDS_11_P]
#set_property PACKAGE_PIN U15 [get_ports $JX1_43_LVDS_10_N]
#set_property PACKAGE_PIN U19 [get_ports $JX1_44_LVDS_11_N]
#set_property PACKAGE_PIN N18 [get_ports $JX1_47_LVDS_12_P]
#set_property PACKAGE_PIN N20 [get_ports $JX1_48_LVDS_13_P]
#set_property PACKAGE_PIN P19 [get_ports $JX1_49_LVDS_12_N]
#set_property PACKAGE_PIN P20 [get_ports $JX1_50_LVDS_13_N]
#set_property PACKAGE_PIN T20 [get_ports $JX1_53_LVDS_14_P]
#set_property PACKAGE_PIN V20 [get_ports $JX1_54_LVDS_15_P]
#set_property PACKAGE_PIN U20 [get_ports $JX1_55_LVDS_14_N]
#set_property PACKAGE_PIN W20 [get_ports $JX1_56_LVDS_15_N]
#set_property PACKAGE_PIN Y18 [get_ports $JX1_61_LVDS_16_P]
#set_property PACKAGE_PIN V16 [get_ports $JX1_62_LVDS_17_P]
#set_property PACKAGE_PIN Y19 [get_ports $JX1_63_LVDS_16_N]
#set_property PACKAGE_PIN W16 [get_ports $JX1_64_LVDS_17_N]
#set_property PACKAGE_PIN R16 [get_ports $JX1_67_LVDS_18_P]
#set_property PACKAGE_PIN T17 [get_ports $JX1_68_LVDS_19_P]
#set_property PACKAGE_PIN R17 [get_ports $JX1_69_LVDS_18_N]
#set_property PACKAGE_PIN R18 [get_ports $JX1_70_LVDS_19_N]
#set_property PACKAGE_PIN V17 [get_ports $JX1_73_LVDS_20_P]
#set_property PACKAGE_PIN W18 [get_ports $JX1_74_LVDS_21_P]
#set_property PACKAGE_PIN V18 [get_ports $JX1_75_LVDS_20_N]
#set_property PACKAGE_PIN W19 [get_ports $JX1_76_LVDS_21_N]
#set_property PACKAGE_PIN N17 [get_ports $JX1_81_LVDS_22_P]
#set_property PACKAGE_PIN P15 [get_ports $JX1_82_LVDS_23_P]
#set_property PACKAGE_PIN P18 [get_ports $JX1_83_LVDS_22_N]
#set_property PACKAGE_PIN P16 [get_ports $JX1_84_LVDS_23_N]


# Set IOSTANDARD
set_property IOSTANDARD LVCMOS25 [get_ports {GPI_IP1[1]}]
set_property IOSTANDARD LVCMOS25 [get_ports {GPI_IP1[0]}]
set_property IOSTANDARD LVCMOS25 [get_ports {GPI_IP0[1]}]
set_property IOSTANDARD LVCMOS25 [get_ports {GPI_IP0[0]}]
set_property IOSTANDARD LVCMOS25 [get_ports {GPO_IP1[1]}]
set_property IOSTANDARD LVCMOS25 [get_ports {GPO_IP1[0]}]
set_property IOSTANDARD LVCMOS25 [get_ports {GPO_IP0[1]}]
set_property IOSTANDARD LVCMOS25 [get_ports {GPO_IP0[0]}]


# =============================================================================
# Pin Constraints: Bank 35
# =============================================================================
# Map the Microzed JX2 connector pins to I/O port names in the design
#   This portion is pinned out to headers on Lotus
set JX2_31_LVDS_4_N     {}
set JX2_35_LVDS_6_P     {}
set JX2_37_LVDS_6_N     {}
set JX2_41_LVDS_8_P     {}
set JX2_42_LVDS_9_P     {}
set JX2_43_LVDS_8_N     {}
set JX2_44_LVDS_9_N     {}
set JX2_47_LVDS_10_P    {}
set JX2_48_LVDS_11_P    {}
set JX2_49_LVDS_10_N    {}
set JX2_50_LVDS_11_N    {}
set JX2_53_LVDS_12_P    {}
set JX2_54_LVDS_13_P    {}
set JX2_55_LVDS_12_N    {}
set JX2_56_LVDS_13_N    {}
set JX2_61_LVDS_14_P    {}
set JX2_62_LVDS_15_P    {}
set JX2_64_LVDS_15_N    {}
set JX2_68_LVDS_17_P    {}
set JX2_70_LVDS_17_N    {}

#   This portion is NOT pinned out to headers on Lotus
set JX2_13_SE_0         {}
set JX2_14_SE_1         {}
set JX2_17_LVDS_0_P     {}
set JX2_18_LVDS_1_P     {}
set JX2_19_LVDS_0_N     {}
set JX2_20_LVDS_1_N     {}
set JX2_23_LVDS_2_P     {}
set JX2_24_LVDS_3_P     {}
set JX2_25_LVDS_2_N     {}
set JX2_26_LVDS_3_N     {}
set JX2_29_LVDS_4_P     {}
set JX2_30_LVDS_5_P     {}
set JX2_32_LVDS_5_N     {}
set JX2_36_LVDS_7_P     {}
set JX2_38_LVDS_7_N     {}
set JX2_63_LVDS_14_N    {}
set JX2_67_LVDS_16_P    {}
set JX2_69_LVDS_16_N    {}
set JX2_73_LVDS_18_P    {}
set JX2_74_LVDS_19_P    {}
set JX2_75_LVDS_18_N    {}
set JX2_76_LVDS_19_N    {}
set JX2_81_LVDS_20_P    {}
set JX2_82_LVDS_21_P    {}
set JX2_83_LVDS_20_N    {}
set JX2_84_LVDS_21_N    {}
set JX2_87_LVDS_22_P    {}
set JX2_88_LVDS_23_P    {}
set JX2_89_LVDS_22_N    {}
set JX2_90_LVDS_23_N    {}


# Uncomment each line per the allocation above
#set_property PACKAGE_PIN G14 [get_ports $JX2_13_SE_0]
#set_property PACKAGE_PIN J15 [get_ports $JX2_14_SE_1]
#set_property PACKAGE_PIN C20 [get_ports $JX2_17_LVDS_0_P]
#set_property PACKAGE_PIN B19 [get_ports $JX2_18_LVDS_1_P]
#set_property PACKAGE_PIN B20 [get_ports $JX2_19_LVDS_0_N]
#set_property PACKAGE_PIN A20 [get_ports $JX2_20_LVDS_1_N]
#set_property PACKAGE_PIN E17 [get_ports $JX2_23_LVDS_2_P]
#set_property PACKAGE_PIN D19 [get_ports $JX2_24_LVDS_3_P]
#set_property PACKAGE_PIN D18 [get_ports $JX2_25_LVDS_2_N]
#set_property PACKAGE_PIN D20 [get_ports $JX2_26_LVDS_3_N]
#set_property PACKAGE_PIN E18 [get_ports $JX2_29_LVDS_4_P]
#set_property PACKAGE_PIN F16 [get_ports $JX2_30_LVDS_5_P]
#set_property PACKAGE_PIN E19 [get_ports $JX2_31_LVDS_4_N]
#set_property PACKAGE_PIN F17 [get_ports $JX2_32_LVDS_5_N]
#set_property PACKAGE_PIN L19 [get_ports $JX2_35_LVDS_6_P]
#set_property PACKAGE_PIN M19 [get_ports $JX2_36_LVDS_7_P]
#set_property PACKAGE_PIN L20 [get_ports $JX2_37_LVDS_6_N]
#set_property PACKAGE_PIN M20 [get_ports $JX2_38_LVDS_7_N]
#set_property PACKAGE_PIN M17 [get_ports $JX2_41_LVDS_8_P]
#set_property PACKAGE_PIN K19 [get_ports $JX2_42_LVDS_9_P]
#set_property PACKAGE_PIN M18 [get_ports $JX2_43_LVDS_8_N]
#set_property PACKAGE_PIN J19 [get_ports $JX2_44_LVDS_9_N]
#set_property PACKAGE_PIN L16 [get_ports $JX2_47_LVDS_10_P]
#set_property PACKAGE_PIN K17 [get_ports $JX2_48_LVDS_11_P]
#set_property PACKAGE_PIN L17 [get_ports $JX2_49_LVDS_10_N]
#set_property PACKAGE_PIN K18 [get_ports $JX2_50_LVDS_11_N]
#set_property PACKAGE_PIN H16 [get_ports $JX2_53_LVDS_12_P]
#set_property PACKAGE_PIN J18 [get_ports $JX2_54_LVDS_13_P]
#set_property PACKAGE_PIN H17 [get_ports $JX2_55_LVDS_12_N]
#set_property PACKAGE_PIN H18 [get_ports $JX2_56_LVDS_13_N]
#set_property PACKAGE_PIN G17 [get_ports $JX2_61_LVDS_14_P]
#set_property PACKAGE_PIN F19 [get_ports $JX2_62_LVDS_15_P]
#set_property PACKAGE_PIN G18 [get_ports $JX2_63_LVDS_14_N]
#set_property PACKAGE_PIN F20 [get_ports $JX2_64_LVDS_15_N]
#set_property PACKAGE_PIN G19 [get_ports $JX2_67_LVDS_16_P]
#set_property PACKAGE_PIN J20 [get_ports $JX2_68_LVDS_17_P]
#set_property PACKAGE_PIN G20 [get_ports $JX2_69_LVDS_16_N]
#set_property PACKAGE_PIN H20 [get_ports $JX2_70_LVDS_17_N]
#set_property PACKAGE_PIN K14 [get_ports $JX2_73_LVDS_18_P]
#set_property PACKAGE_PIN H15 [get_ports $JX2_74_LVDS_19_P]
#set_property PACKAGE_PIN J14 [get_ports $JX2_75_LVDS_18_N]
#set_property PACKAGE_PIN G15 [get_ports $JX2_76_LVDS_19_N]
#set_property PACKAGE_PIN N15 [get_ports $JX2_81_LVDS_20_P]
#set_property PACKAGE_PIN L14 [get_ports $JX2_82_LVDS_21_P]
#set_property PACKAGE_PIN N16 [get_ports $JX2_83_LVDS_20_N]
#set_property PACKAGE_PIN L15 [get_ports $JX2_84_LVDS_21_N]
#set_property PACKAGE_PIN M14 [get_ports $JX2_87_LVDS_22_P]
#set_property PACKAGE_PIN K16 [get_ports $JX2_88_LVDS_23_P]
#set_property PACKAGE_PIN M15 [get_ports $JX2_89_LVDS_22_N]
#set_property PACKAGE_PIN J16 [get_ports $JX2_90_LVDS_23_N]


# Set IOSTANDARD
#set_property IOSTANDARD LVCMOS33 [get_ports {GPI_IP1[1]}]
#set_property IOSTANDARD LVCMOS33 [get_ports {GPI_IP1[0]}]
#set_property IOSTANDARD LVCMOS33 [get_ports {GPI_IP0[1]}]
#set_property IOSTANDARD LVCMOS33 [get_ports {GPI_IP0[0]}]
#set_property IOSTANDARD LVCMOS33 [get_ports {GPO_IP1[1]}]
#set_property IOSTANDARD LVCMOS33 [get_ports {GPO_IP1[0]}]
#set_property IOSTANDARD LVCMOS33 [get_ports {GPO_IP0[1]}]
#set_property IOSTANDARD LVCMOS33 [get_ports {GPO_IP0[0]}]
