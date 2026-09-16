`include "lotus_defines.v"
`timescale 1 ns / 1 ps

module lotus_output_drivers (
    input   wire                                        OEN,

    // mini-lvds
    input   wire                                        LVDS_CLK,
    input   wire    [5 : 0]                             LVDS_DATA,
    input   wire                                        POL_INT,
    input   wire                                        TP1_INT,

    output  wire                                        LVDS_0N,
    output  wire                                        LVDS_0P,
    output  wire                                        LVDS_1N,
    output  wire                                        LVDS_1P,
    output  wire                                        LVDS_2N,
    output  wire                                        LVDS_2P,
    output  wire                                        LVDS_3N,
    output  wire                                        LVDS_3P,
    output  wire                                        LVDS_4N,
    output  wire                                        LVDS_4P,
    output  wire                                        LVDS_5N,
    output  wire                                        LVDS_5P,
    output  wire                                        LVDS_CLK_N,
    output  wire                                        LVDS_CLK_P,
    output  wire                                        POL,
    output  wire                                        TP1,

    // GPO
    input   wire                                        ITO_CLK_INT,
    input   wire                                        STATE_EQ_PROG_INT,

    output  wire                                        ITO_CLK,
    output  wire                                        STATE_EQ_PROG,

    // Laser control
    input   wire                                        LASER_DR1_INT,
    input   wire                                        LASER_DR2_INT,
    input   wire                                        LASER_TRIGGER_INT,

    output  wire                                        LASER_DR1_N,
    output  wire                                        LASER_DR1_P,
    output  wire                                        LASER_DR2_N,
    output  wire                                        LASER_DR2_P,
    output  wire                                        LASER_TRIGGER
);


    // ------------------------------------------------------------------------
    // mini-lvds
    // ------------------------------------------------------------------------
    // LVDS_DATA
    wire [5 : 0] lvds_n;
    wire [5 : 0] lvds_p;

    assign {LVDS_5N, LVDS_4N, LVDS_3N, LVDS_2N, LVDS_1N, LVDS_0N} = lvds_n;
    assign {LVDS_5P, LVDS_4P, LVDS_3P, LVDS_2P, LVDS_1P, LVDS_0P} = lvds_p;

    genvar i_gv;

    generate
        for( i_gv = 0; i_gv < 6; i_gv = i_gv + 1 )
        begin : gen_obuftds
            OBUFTDS #(
                .IOSTANDARD ("MINI_LVDS_25"))
            I_OBUFTDS_LVDS(
                .I          (LVDS_DATA[i_gv]),
                .T          (~OEN),

                .O          (lvds_p[i_gv]),
                .OB         (lvds_n[i_gv])
            );
        end
    endgenerate


    // LVDS_CLK
    OBUFTDS #(
        .IOSTANDARD ("MINI_LVDS_25"))
    I_OBUFTDS_LVDS_CLK(
        .I          (LVDS_CLK),
        .T          (~OEN),

        .O          (LVDS_CLK_P),
        .OB         (LVDS_CLK_N)
    );


    // POL
    `ifdef ZYNQ_CARRIER_CARD
    OBUFT #(
        .IOSTANDARD ("LVCMOS25"),
    `else
    OBUFT #(
        .IOSTANDARD ("LVCMOS33"),
    `endif
        .DRIVE      (12),       // 4, 8, 12, or 16 mA
        .SLEW       ("SLOW"))   // "SLOW" or "FAST"
    I_OBUFT_POL(
        .I          (POL_INT),
        .T          (~OEN),
        .O          (POL)
    );


    // TP1
    `ifdef ZYNQ_CARRIER_CARD
    OBUFT #(
        .IOSTANDARD ("LVCMOS25"),
    `else
    OBUFT #(
        .IOSTANDARD ("LVCMOS33"),
    `endif
        .DRIVE      (12),       // 4, 8, 12, or 16 mA
        .SLEW       ("SLOW"))   // "SLOW" or "FAST"
    I_OBUFT_TP1(
        .I          (TP1_INT),
        .T          (~OEN),
        .O          (TP1)
    );


    // ------------------------------------------------------------------------
    // GPO
    // ------------------------------------------------------------------------
    // ITO_CLK
    `ifdef ZYNQ_CARRIER_CARD
    OBUFT #(
        .IOSTANDARD ("LVCMOS25"),
    `else
    OBUFT #(
        .IOSTANDARD ("LVCMOS33"),
    `endif
        .DRIVE      (12),       // 4, 8, 12, or 16 mA
        .SLEW       ("SLOW"))   // "SLOW" or "FAST"
    I_OBUFT_ITO_CLK(
        .I          (ITO_CLK_INT),
        .T          (~OEN),
        .O          (ITO_CLK)
    );

    `ifdef ZYNQ_CARRIER_CARD
    OBUFT #(
        .IOSTANDARD ("LVCMOS25"),
    `else
    OBUFT #(
        .IOSTANDARD ("LVCMOS33"),
    `endif
        .DRIVE      (12),       // 4, 8, 12, or 16 mA
        .SLEW       ("SLOW"))   // "SLOW" or "FAST"
    I_OBUFT_STATE_EQ_PROG(
        .I          (STATE_EQ_PROG_INT),
        .T          (~OEN),
        .O          (STATE_EQ_PROG)
    );


    // ------------------------------------------------------------------------
    // Laser control
    // ------------------------------------------------------------------------
    // LASER_DR1
    OBUFTDS #(
        .IOSTANDARD ("LVDS_25"))
    I_OBUFTDS_LASER_DR1(
        .I          (LASER_DR1_INT),
        .T          (~OEN),

        .O          (LASER_DR1_P),
        .OB         (LASER_DR1_N)
    );

    // LASER_DR2
    OBUFTDS #(
        .IOSTANDARD ("LVDS_25"))
    I_OBUFTDS_LASER_DR2(
        .I          (LASER_DR2_INT),
        .T          (~OEN),

        .O          (LASER_DR2_P),
        .OB         (LASER_DR2_N)
    );

    // LASER_TRIGGER
    `ifdef ZYNQ_CARRIER_CARD
    OBUFT #(
        .IOSTANDARD ("LVCMOS25"),
    `else
    OBUFT #(
        .IOSTANDARD ("LVCMOS33"),
    `endif
        .DRIVE      (12),       // 4, 8, 12, or 16 mA
        .SLEW       ("SLOW"))   // "SLOW" or "FAST"
    I_OBUFT_LASER_TRIGGER(
        .I          (LASER_TRIGGER_INT),
        .T          (~OEN),
        .O          (LASER_TRIGGER)
    );

 endmodule
