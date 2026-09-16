`include "bcuda_defines.v"
`timescale 1 ns / 1 ps

// A wrapper for eight instances of the matched filter.

module bcuda_mf #(
    parameter integer N_TAPS = 16,
    parameter integer H_N_BITS = 16,
    parameter integer X_N_BITS = 16,
    parameter integer Y_N_BITS = 16
)
(
    input   wire                                        CLK,
    input   wire                                        RESET,
    input   wire        [H_N_BITS * N_TAPS - 1 : 0]     H,
    input   wire signed [X_N_BITS - 1 : 0]              X0,
    input   wire                                        X0_DV,
    input   wire signed [X_N_BITS - 1 : 0]              X1,
    input   wire                                        X1_DV,
    input   wire signed [X_N_BITS - 1 : 0]              X2,
    input   wire                                        X2_DV,
    input   wire signed [X_N_BITS - 1 : 0]              X3,
    input   wire                                        X3_DV,
    input   wire signed [X_N_BITS - 1 : 0]              X4,
    input   wire                                        X4_DV,
    input   wire signed [X_N_BITS - 1 : 0]              X5,
    input   wire                                        X5_DV,
    input   wire signed [X_N_BITS - 1 : 0]              X6,
    input   wire                                        X6_DV,
    input   wire signed [X_N_BITS - 1 : 0]              X7,
    input   wire                                        X7_DV,

    output  wire signed [Y_N_BITS - 1 : 0]              Y0,
    output  wire                                        Y0_DV,
    output  wire signed [Y_N_BITS - 1 : 0]              Y1,
    output  wire                                        Y1_DV,
    output  wire signed [Y_N_BITS - 1 : 0]              Y2,
    output  wire                                        Y2_DV,
    output  wire signed [Y_N_BITS - 1 : 0]              Y3,
    output  wire                                        Y3_DV,
    output  wire signed [Y_N_BITS - 1 : 0]              Y4,
    output  wire                                        Y4_DV,
    output  wire signed [Y_N_BITS - 1 : 0]              Y5,
    output  wire                                        Y5_DV,
    output  wire signed [Y_N_BITS - 1 : 0]              Y6,
    output  wire                                        Y6_DV,
    output  wire signed [Y_N_BITS - 1 : 0]              Y7,
    output  wire                                        Y7_DV
);


    localparam integer SLICE_Y_N_BITS = X_N_BITS + H_N_BITS - 1;

    genvar i_gv;

    wire signed [X_N_BITS - 1 : 0] fir_x    [7 : 0];
    wire                           fir_x_dv [7 : 0];
    wire signed [Y_N_BITS - 1 : 0] fir_y    [7 : 0];
    wire                           fir_y_dv [7 : 0];

    assign fir_x[0] = X0;
    assign fir_x[1] = X1;
    assign fir_x[2] = X2;
    assign fir_x[3] = X3;
    assign fir_x[4] = X4;
    assign fir_x[5] = X5;
    assign fir_x[6] = X6;
    assign fir_x[7] = X7;
    assign fir_x_dv[0] = X0_DV;
    assign fir_x_dv[1] = X1_DV;
    assign fir_x_dv[2] = X2_DV;
    assign fir_x_dv[3] = X3_DV;
    assign fir_x_dv[4] = X4_DV;
    assign fir_x_dv[5] = X5_DV;
    assign fir_x_dv[6] = X6_DV;
    assign fir_x_dv[7] = X7_DV;

    assign Y0 = fir_y[0];
    assign Y1 = fir_y[1];
    assign Y2 = fir_y[2];
    assign Y3 = fir_y[3];
    assign Y4 = fir_y[4];
    assign Y5 = fir_y[5];
    assign Y6 = fir_y[6];
    assign Y7 = fir_y[7];
    assign Y0_DV = fir_y_dv[0];
    assign Y1_DV = fir_y_dv[1];
    assign Y2_DV = fir_y_dv[2];
    assign Y3_DV = fir_y_dv[3];
    assign Y4_DV = fir_y_dv[4];
    assign Y5_DV = fir_y_dv[5];
    assign Y6_DV = fir_y_dv[6];
    assign Y7_DV = fir_y_dv[7];

    generate
        for (i_gv = 0; i_gv < 8; i_gv = i_gv + 1)
        begin : gen_mf
            bcuda_mf_fir #(
                .N_TAPS                 (N_TAPS),
                .H_N_BITS               (H_N_BITS),
                .X_N_BITS               (X_N_BITS),
                .Y_SLICE_N_BITS         (SLICE_Y_N_BITS),
                .Y_N_BITS               (Y_N_BITS))
            i_mf_fir (
                .CLK                    (CLK),
                .RESET                  (RESET),
                .H                      (H),
                .X                      (fir_x[i_gv]),
                .X_DV                   (fir_x_dv[i_gv]),
                .Y                      (fir_y[i_gv]),
                .Y_DV                   (fir_y_dv[i_gv])
            );
        end
    endgenerate

endmodule
