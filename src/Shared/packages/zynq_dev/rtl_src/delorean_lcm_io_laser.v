`include "delorean_defines.v"
`timescale 1 ns / 1 ps

module delorean_lcm_io_laser (
    input   wire                                        OUT_EN,
    input   wire                                        LASER_DR1_INT,
    input   wire                                        LASER_DR2_INT,
    input   wire                                        LASER_TRIGGER_INT,
    input   wire                                        TX_PWR_EN_INT,
    input   wire                                        TX_PWR_SWITCH_INT,

    output  wire    [1 : 0]                             LASER_DR1_P,
    output  wire    [1 : 0]                             LASER_DR1_N,
    output  wire    [1 : 0]                             LASER_DR2_P,
    output  wire    [1 : 0]                             LASER_DR2_N,
    output  wire                                        LASER_TRIGGER,
    output  wire                                        TX_PWR_EN,
    output  wire                                        TX_PWR_SWITCH
);


    // LASER_DR1
    OBUFTDS #(
        .IOSTANDARD     ("LVDS_25"))
    U_OBUFTDS_LASER_DR1[1 : 0] (
        .I              (LASER_DR1_INT),
        .T              (~OUT_EN),          // active low
        .O              (LASER_DR1_P),
        .OB             (LASER_DR1_N)
    );


    // LASER_DR2
    OBUFTDS #(
        .IOSTANDARD     ("LVDS_25"))
    U_OBUFTDS_LASER_DR2[1 : 0] (
        .I              (LASER_DR2_INT),
        .T              (~OUT_EN),          // active low
        .O              (LASER_DR2_P),
        .OB             (LASER_DR2_N)
    );


    // LASER_TRIGGER
    OBUFT #(
        .IOSTANDARD     ("LVCMOS33"),
        .DRIVE          (12),               // 4, 8, 12, or 16 mA
        .SLEW           ("SLOW"))           // "SLOW" or "FAST"
    I_OBUFT_LASER_TRIGGER(
        .I              (LASER_TRIGGER_INT),
        .T              (~OUT_EN),
        .O              (LASER_TRIGGER)
    );


    // TX_PWR_EN
    OBUFT #(
        .IOSTANDARD     ("LVCMOS33"),
        .DRIVE          (12),               // 4, 8, 12, or 16 mA
        .SLEW           ("SLOW"))           // "SLOW" or "FAST"
    U_OBUFT_TX_PWR_EN (
        .I              (TX_PWR_EN_INT),
        .T              (~OUT_EN),
        .O              (TX_PWR_EN)
    );

    // TX_PWR_SWITCH
    OBUFT #(
        .IOSTANDARD     ("LVCMOS33"),
        .DRIVE          (12),               // 4, 8, 12, or 16 mA
        .SLEW           ("SLOW"))           // "SLOW" or "FAST"
    U_OBUFT_TX_PWR_SWITCH (
        .I              (TX_PWR_SWITCH_INT),
        .T              (~OUT_EN),
        .O              (TX_PWR_SWITCH)
    );

endmodule
