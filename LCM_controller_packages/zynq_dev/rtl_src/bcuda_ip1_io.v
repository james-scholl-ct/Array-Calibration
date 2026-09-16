`include "bcuda_defines.v"
`timescale 1 ns / 1 ps

module bcuda_ip1_io #(
    parameter integer BYPASS = 0,
    parameter integer GPI_N_BITS = 2,
    parameter integer GPO_N_BITS = 2
)
(
    input   wire                                        OEN,

    // fabric side
    input   wire    [GPO_N_BITS - 1 : 0]                GPO_INT,

    output  wire    [GPI_N_BITS - 1 : 0]                GPI_INT,

    // pad side
    input   wire    [GPI_N_BITS - 1 : 0]                GPI,

    output  wire    [GPO_N_BITS - 1 : 0]                GPO
);


    // ------------------------------------------------------------------------
    // GPIO
    // ------------------------------------------------------------------------
    generate
        if (BYPASS)
        begin
            assign GPI_INT = GPI;
            assign GPO = ~OEN ? GPO_INT : {GPO_N_BITS{1'bz}};
        end
        else
        begin
            IBUF #(
                .IOSTANDARD ("LVCMOS25"))
            i_IBUF_gpi[GPI_N_BITS - 1 : 0](
                .I          (GPI),
                .O          (GPI_INT)
            );

            OBUFT #(
                .IOSTANDARD ("LVCMOS25"),
                .DRIVE      (16),           // 4, 8, 12, or 16 mA
                .SLEW       ("FAST"))       // "SLOW" or "FAST"
            i_OBUFT_gpo[GPI_N_BITS - 1 : 0](
                .I          (GPO_INT),
                .T          (~OEN),
                .O          (GPO)
            );
        end
    endgenerate


endmodule
