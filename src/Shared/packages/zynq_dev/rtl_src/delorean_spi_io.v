`include "delorean_defines.v"
`timescale 1 ns / 1 ps

module delorean_spi_io #(
    parameter integer N_SLAVES = 8
)
(
    input   wire                                        OEN,

    input   wire    [1 : 0]                             ADC_CHSEL_INT,
    input   wire                                        DAISY_EN_INT,
    input   wire    [1 : 0]                             MOSI_INT,
    input   wire                                        SCLK_INT,
    input   wire    [N_SLAVES - 1 : 0]                  SS_B_INT,
    output  wire    [1 : 0]                             MISO_INT,

    input   wire    [1 : 0]                             MISO,
    output  wire    [1 : 0]                             ADC_CHSEL,
    output  wire                                        DAISY_EN,
    output  wire    [1 : 0]                             MOSI,
    output  wire    [1 : 0]                             SCLK,
    output  wire    [N_SLAVES - 1 : 0]                  SS_B
);


    // ------------------------------------------------------------------------
    // MISO
    // ------------------------------------------------------------------------
    IBUF #(
        .IOSTANDARD ("LVCMOS33"))
    I_IBUF_MISO[1:0](
        .I          (MISO),
        .O          (MISO_INT)
    );

    // ------------------------------------------------------------------------
    // ADC_CHSEL
    // ------------------------------------------------------------------------
    OBUFT #(
        .IOSTANDARD ("LVCMOS33"),
        .DRIVE      (16),       // 4, 8, 12, or 16 mA
        .SLEW       ("SLOW"))   // "SLOW" or "FAST"
    I_OBUFT_ADC_CHSEL[1:0](
        .I          (ADC_CHSEL_INT),
        .T          (~OEN),
        .O          (ADC_CHSEL)
    );

    // ------------------------------------------------------------------------
    // DAISY_EN
    // ------------------------------------------------------------------------
    OBUFT #(
        .IOSTANDARD ("LVCMOS33"),
        .DRIVE      (16),       // 4, 8, 12, or 16 mA
        .SLEW       ("SLOW"))   // "SLOW" or "FAST"
    I_OBUFT_DAISY_EN(
        .I          (DAISY_EN_INT),
        .T          (~OEN),
        .O          (DAISY_EN)
    );

    // ------------------------------------------------------------------------
    // MOSI
    // ------------------------------------------------------------------------
    OBUFT #(
        .IOSTANDARD ("LVCMOS33"),
        .DRIVE      (16),       // 4, 8, 12, or 16 mA
        .SLEW       ("FAST"))   // "SLOW" or "FAST"
    I_OBUFT_MOSI[1:0](
        .I          (MOSI_INT),
        .T          (~OEN),
        .O          (MOSI)
    );

    // ------------------------------------------------------------------------
    // SCLK
    // ------------------------------------------------------------------------
    OBUFT #(
        .IOSTANDARD ("LVCMOS33"),
        .DRIVE      (16),       // 4, 8, 12, or 16 mA
        .SLEW       ("FAST"))   // "SLOW" or "FAST"
    I_OBUFT_SCLK[1 : 0](
        .I          (SCLK_INT),
        .T          (~OEN),
        .O          (SCLK)
    );

    // ------------------------------------------------------------------------
    // SS_B
    // ------------------------------------------------------------------------
    OBUFT #(
        .IOSTANDARD ("LVCMOS33"),
        .DRIVE      (16),       // 4, 8, 12, or 16 mA
        .SLEW       ("FAST"))   // "SLOW" or "FAST"
    I_OBUFT_SS[N_SLAVES - 1 : 0](
        .I          (SS_B_INT),
        .T          (~OEN),
        .O          (SS_B)
    );

 endmodule
