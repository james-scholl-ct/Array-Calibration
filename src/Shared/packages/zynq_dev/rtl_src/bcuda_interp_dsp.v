`include "bcuda_defines.v"
`timescale 1 ns / 1 ps

module bcuda_interp_dsp #(
    parameter integer X_N_BITS = 16,
    parameter integer H_N_BITS = 16,
    parameter integer Y_N_BITS = 32
)
(
    input   wire                                        CLK,
    input   wire                                        RESET,
    input   wire signed [H_N_BITS - 1 : 0]              H,
    input   wire                                        X_DV,
    input   wire signed [X_N_BITS - 1 : 0]              X_TAP0,
    input   wire signed [X_N_BITS - 1 : 0]              X_TAP1,
    input   wire signed [Y_N_BITS - 1 : 0]              X_ACC,

    output  reg  signed [Y_N_BITS - 1 : 0]              Y
);


    localparam integer SUM_N_BITS = X_N_BITS + 1;

    reg  signed [SUM_N_BITS - 1 : 0] sum;
    reg  signed [H_N_BITS - 1 : 0]   h_dly;
    reg  signed [Y_N_BITS - 1 : 0]   prod;

    always @(posedge CLK)
    begin
        if (RESET)
        begin
            sum     <= 0;
            h_dly   <= 0;
            prod    <= 0;
            Y       <= 0;
        end
        else if (X_DV)
        begin
            sum     <= X_TAP0 + X_TAP1;
            h_dly   <= H;
            prod    <= sum * h_dly;
            Y       <= prod + X_ACC;
        end
    end

endmodule
