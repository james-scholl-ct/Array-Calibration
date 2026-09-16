`include "delorean_defines.v"
`timescale 1 ns / 1 ps

module delorean_lcm_io_mlvds (
    input   wire                                        CLK,
    input   wire                                        CLK_90,

    // Mini-LVDS
    input   wire    [5 : 0]                             LVDS_DATA_D1,
    input   wire    [5 : 0]                             LVDS_DATA_D2,
    input   wire                                        POL_INT,
    input   wire                                        TP1_INT,

    output  wire                                        LVDS_CLK_P,
    output  wire                                        LVDS_CLK_N,
    output  wire    [5 : 0]                             LVDS_DATA_P,
    output  wire    [5 : 0]                             LVDS_DATA_N,
    output  wire                                        POL,
    output  wire                                        TP1,

    // GPO
    input   wire                                        ITO_CLK_INT,
    input   wire                                        LCD_EN_INT,
    input   wire                                        PROG_TRIGGER_INT,

    output  wire                                        ITO_CLK,
    output  wire                                        LCD_EN,
    output  wire                                        PROG_TRIGGER
);

    //wire out_en_b = ~LCD_EN_INT;
    wire out_en_b = 1'b0;

    // ------------------------------------------------------------------------
    // Mini-LVDS
    // ------------------------------------------------------------------------
    // LVDS_DATA
    wire [5 : 0] lvds_data;
    wire [5 : 0] lvds_data_en_b;

    // - OLOGIC: data path
    ODDR #(
        .DDR_CLK_EDGE   ("SAME_EDGE"),
        .INIT           (1'b0),
        .SRTYPE         ("SYNC"))
    U_ODDR_LVDS_DATA[5 : 0] (
        .C              (CLK),
        .R              (1'b0),
        .S              (1'b0),
        .CE             (1'b1),
        .D1             (LVDS_DATA_D1),
        .D2             (LVDS_DATA_D2),
        .Q              (lvds_data)
    );

    // - OLOGIC: control path (pipelined/forwarded)
    ODDR #(
        .DDR_CLK_EDGE   ("SAME_EDGE"),
        .INIT           (1'b0),
        .SRTYPE         ("SYNC"))
    U_ODDR_LVDS_DATA_EN[5 : 0] (
        .C              (CLK),
        .R              (1'b0),
        .S              (1'b0),
        .CE             (1'b1),
        .D1             (out_en_b),
        .D2             (out_en_b),
        .Q              (lvds_data_en_b)
    );

    // - IOB
    OBUFTDS #(
        .IOSTANDARD     ("MINI_LVDS_25"))
    U_OBUFTDS_LVDS_DATA[5 : 0] (
        .I              (lvds_data),
        .T              (lvds_data_en_b),   // active low
        .O              (LVDS_DATA_P),
        .OB             (LVDS_DATA_N)
    );


    // LVDS_CLK
    wire lvds_clk;

    // - OLOGIC (for clock forwarding)
    ODDR #(
        .DDR_CLK_EDGE   ("SAME_EDGE"),
        .INIT           (1'b0),
        .SRTYPE         ("SYNC"))
    U_ODDR_LVDS_CLK (
        .C              (CLK_90),
        .R              (1'b0),
        .S              (1'b0),
        .CE             (1'b1),
        .D1             (1'b1),
        .D2             (1'b0),
        .Q              (lvds_clk)
    );

    // - IOB
    OBUFTDS #(
        .IOSTANDARD     ("MINI_LVDS_25"))
    U_OBUFTDS_LVDS_CLK (
        .I              (lvds_clk),
        .T              (out_en_b),     // active low
        .O              (LVDS_CLK_P),
        .OB             (LVDS_CLK_N)
    );


    // POL
    OBUFT #(
        .IOSTANDARD     ("LVCMOS33"),
        .DRIVE          (12),           // 4, 8, 12, or 16 mA
        .SLEW           ("SLOW"))       // "SLOW" or "FAST"
    U_OBUFT_POL(
        .I              (POL_INT),
        .T              (out_en_b),     // active low
        .O              (POL)
    );


    // TP1
    OBUFT #(
        .IOSTANDARD     ("LVCMOS33"),
        .DRIVE          (12),           // 4, 8, 12, or 16 mA
        .SLEW           ("SLOW"))       // "SLOW" or "FAST"
    U_OBUFT_TP1(
        .I              (TP1_INT),
        .T              (out_en_b),     // active low
        .O              (TP1)
    );

    // ------------------------------------------------------------------------
    // GPO
    // ------------------------------------------------------------------------
    // ITO_CLK
    OBUFT #(
        .IOSTANDARD     ("LVCMOS33"),
        .DRIVE          (12),           // 4, 8, 12, or 16 mA
        .SLEW           ("SLOW"))       // "SLOW" or "FAST"
    U_OBUFT_ITO_CLK(
        .I              (ITO_CLK_INT),
        .T              (out_en_b),     // active low
        .O              (ITO_CLK)
    );


    // LCD_EN
    OBUF #(
        .IOSTANDARD     ("LVCMOS33"),
        .DRIVE          (12),           // 4, 8, 12, or 16 mA
        .SLEW           ("SLOW"))       // "SLOW" or "FAST"
    U_OBUF_LCD_EN(
        .I              (LCD_EN_INT),
        .O              (LCD_EN)
    );


    // PROG_TRIGGER
    OBUFT #(
        .IOSTANDARD     ("LVCMOS33"),
        .DRIVE          (12),           // 4, 8, 12, or 16 mA
        .SLEW           ("SLOW"))       // "SLOW" or "FAST"
    U_OBUFT_PROG_TRIGGER(
        .I              (PROG_TRIGGER_INT),
        .T              (out_en_b),     // active low
        .O              (PROG_TRIGGER)
    );

 endmodule
