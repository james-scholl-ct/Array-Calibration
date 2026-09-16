`include "bcuda_defines.v"
`timescale 1 ns / 1 ps

module bcuda_interp #(
    parameter integer X_N_BITS = 16,
    parameter integer FIR0_H_N_BITS = 16,
    parameter integer FIR0_Y_N_BITS = 16,
    parameter integer FIR1_H_N_BITS = 16,
    parameter integer FIR1_Y_N_BITS = 16,
    parameter integer FIR2_H_N_BITS = 16,
    parameter integer FIR2_Y_N_BITS = 16
)
(
    input   wire                                        CLK,
    input   wire                                        RESET,
    input   wire        [9 * FIR0_H_N_BITS - 1 : 0]     H0,
    input   wire        [5 * FIR1_H_N_BITS - 1 : 0]     H1,
    input   wire        [5 * FIR2_H_N_BITS - 1 : 0]     H2,
    input   wire signed [X_N_BITS - 1 : 0]              X,
    input   wire                                        X_DV,

    output  wire signed [FIR2_Y_N_BITS - 1 : 0]         Y0,
    output  wire                                        Y0_DV,
    output  wire signed [FIR2_Y_N_BITS - 1 : 0]         Y1,
    output  wire                                        Y1_DV,
    output  wire signed [FIR2_Y_N_BITS - 1 : 0]         Y2,
    output  wire                                        Y2_DV,
    output  wire signed [FIR2_Y_N_BITS - 1 : 0]         Y3,
    output  wire                                        Y3_DV,
    output  wire signed [FIR2_Y_N_BITS - 1 : 0]         Y4,
    output  wire                                        Y4_DV,
    output  wire signed [FIR2_Y_N_BITS - 1 : 0]         Y5,
    output  wire                                        Y5_DV,
    output  wire signed [FIR2_Y_N_BITS - 1 : 0]         Y6,
    output  wire                                        Y6_DV,
    output  wire signed [FIR2_Y_N_BITS - 1 : 0]         Y7,
    output  wire                                        Y7_DV
);


    localparam integer FIR0_N_TAPS = 31;
    localparam integer FIR1_N_TAPS = 15;
    localparam integer FIR2_N_TAPS = 15;

    // Add one due to L1_norm on [1, 2).
    localparam integer FIR0_SLICE_Y_N_BITS = (X_N_BITS + FIR0_H_N_BITS - 1) + 1;
    localparam integer FIR1_SLICE_Y_N_BITS = (FIR0_Y_N_BITS + FIR1_H_N_BITS - 1) + 1;
    localparam integer FIR2_SLICE_Y_N_BITS = (FIR1_Y_N_BITS + FIR2_H_N_BITS - 1) + 1;


    wire signed [FIR0_Y_N_BITS - 1 : 0] fir0_y0;
    wire signed [FIR0_Y_N_BITS - 1 : 0] fir0_y1;
    wire                                fir0_y0_dv;
    wire                                fir0_y1_dv;

    wire signed [FIR1_Y_N_BITS - 1 : 0] fir1_y0;
    wire signed [FIR1_Y_N_BITS - 1 : 0] fir1_y1;
    wire signed [FIR1_Y_N_BITS - 1 : 0] fir1_y2;
    wire signed [FIR1_Y_N_BITS - 1 : 0] fir1_y3;
    wire                                fir1_y0_dv;
    wire                                fir1_y1_dv;
    wire                                fir1_y2_dv;
    wire                                fir1_y3_dv;

    wire signed [FIR2_Y_N_BITS - 1 : 0] fir2_y0;
    wire signed [FIR2_Y_N_BITS - 1 : 0] fir2_y1;
    wire signed [FIR2_Y_N_BITS - 1 : 0] fir2_y2;
    wire signed [FIR2_Y_N_BITS - 1 : 0] fir2_y3;
    wire signed [FIR2_Y_N_BITS - 1 : 0] fir2_y4;
    wire signed [FIR2_Y_N_BITS - 1 : 0] fir2_y5;
    wire signed [FIR2_Y_N_BITS - 1 : 0] fir2_y6;
    wire signed [FIR2_Y_N_BITS - 1 : 0] fir2_y7;
    wire                                fir2_y0_dv;
    wire                                fir2_y1_dv;
    wire                                fir2_y2_dv;
    wire                                fir2_y3_dv;
    wire                                fir2_y4_dv;
    wire                                fir2_y5_dv;
    wire                                fir2_y6_dv;
    wire                                fir2_y7_dv;


    // Clip negative values ahead of filtering
    wire signed [X_N_BITS - 1 : 0]  x_clip;

    assign x_clip = (X < 0) ? 0 : X;


    // First stage
    bcuda_interp_fir #(
        .N_TAPS                 (FIR0_N_TAPS),
        .N_THREADS              (8),
        .H_N_BITS               (FIR0_H_N_BITS),
        .X_N_BITS               (X_N_BITS),
        .Y_SLICE_N_BITS         (FIR0_SLICE_Y_N_BITS),
        .Y_N_BITS               (FIR0_Y_N_BITS))
    i_interp_fir00 (
        .CLK                    (CLK),
        .RESET                  (RESET),
        .H                      (H0),
        .X                      (x_clip),
        .X_DV                   (X_DV),
        .Y0                     (fir0_y0),
        .Y0_DV                  (fir0_y0_dv),
        .Y1                     (fir0_y1),
        .Y1_DV                  (fir0_y1_dv)
    );

    // Second stage
    bcuda_interp_fir #(
        .N_TAPS                 (FIR1_N_TAPS),
        .N_THREADS              (4),
        .H_N_BITS               (FIR1_H_N_BITS),
        .X_N_BITS               (FIR0_Y_N_BITS),
        .Y_SLICE_N_BITS         (FIR1_SLICE_Y_N_BITS),
        .Y_N_BITS               (FIR1_Y_N_BITS))
    i_interp_fir10 (
        .CLK                    (CLK),
        .RESET                  (RESET),
        .H                      (H1),
        .X                      (fir0_y0),
        .X_DV                   (fir0_y0_dv),
        .Y0                     (fir1_y0),
        .Y0_DV                  (fir1_y0_dv),
        .Y1                     (fir1_y1),
        .Y1_DV                  (fir1_y1_dv)
    );

    bcuda_interp_fir #(
        .N_TAPS                 (FIR1_N_TAPS),
        .N_THREADS              (4),
        .H_N_BITS               (FIR1_H_N_BITS),
        .X_N_BITS               (FIR0_Y_N_BITS),
        .Y_SLICE_N_BITS         (FIR1_SLICE_Y_N_BITS),
        .Y_N_BITS               (FIR1_Y_N_BITS))
    i_interp_fir11 (
        .CLK                    (CLK),
        .RESET                  (RESET),
        .H                      (H1),
        .X                      (fir0_y1),
        .X_DV                   (fir0_y1_dv),
        .Y0                     (fir1_y2),
        .Y0_DV                  (fir1_y2_dv),
        .Y1                     (fir1_y3),
        .Y1_DV                  (fir1_y3_dv)
    );

    // Third stage
    bcuda_interp_fir #(
        .N_TAPS                 (FIR2_N_TAPS),
        .N_THREADS              (2),
        .H_N_BITS               (FIR2_H_N_BITS),
        .X_N_BITS               (FIR1_Y_N_BITS),
        .Y_SLICE_N_BITS         (FIR2_SLICE_Y_N_BITS),
        .Y_N_BITS               (FIR2_Y_N_BITS))
    i_interp_fir20 (
        .CLK                    (CLK),
        .RESET                  (RESET),
        .H                      (H2),
        .X                      (fir1_y0),
        .X_DV                   (fir1_y0_dv),
        .Y0                     (fir2_y0),
        .Y0_DV                  (fir2_y0_dv),
        .Y1                     (fir2_y1),
        .Y1_DV                  (fir2_y1_dv)
    );

    bcuda_interp_fir #(
        .N_TAPS                 (FIR2_N_TAPS),
        .N_THREADS              (2),
        .H_N_BITS               (FIR2_H_N_BITS),
        .X_N_BITS               (FIR1_Y_N_BITS),
        .Y_SLICE_N_BITS         (FIR2_SLICE_Y_N_BITS),
        .Y_N_BITS               (FIR2_Y_N_BITS))
    i_interp_fir21 (
        .CLK                    (CLK),
        .RESET                  (RESET),
        .H                      (H2),
        .X                      (fir1_y1),
        .X_DV                   (fir1_y1_dv),
        .Y0                     (fir2_y2),
        .Y0_DV                  (fir2_y2_dv),
        .Y1                     (fir2_y3),
        .Y1_DV                  (fir2_y3_dv)
    );

    bcuda_interp_fir #(
        .N_TAPS                 (FIR2_N_TAPS),
        .N_THREADS              (2),
        .H_N_BITS               (FIR2_H_N_BITS),
        .X_N_BITS               (FIR1_Y_N_BITS),
        .Y_SLICE_N_BITS         (FIR2_SLICE_Y_N_BITS),
        .Y_N_BITS               (FIR2_Y_N_BITS))
    i_interp_fir22 (
        .CLK                    (CLK),
        .RESET                  (RESET),
        .H                      (H2),
        .X                      (fir1_y2),
        .X_DV                   (fir1_y2_dv),
        .Y0                     (fir2_y4),
        .Y0_DV                  (fir2_y4_dv),
        .Y1                     (fir2_y5),
        .Y1_DV                  (fir2_y5_dv)
    );

    bcuda_interp_fir #(
        .N_TAPS                 (FIR2_N_TAPS),
        .N_THREADS              (2),
        .H_N_BITS               (FIR2_H_N_BITS),
        .X_N_BITS               (FIR1_Y_N_BITS),
        .Y_SLICE_N_BITS         (FIR2_SLICE_Y_N_BITS),
        .Y_N_BITS               (FIR2_Y_N_BITS))
    i_interp_fir23 (
        .CLK                    (CLK),
        .RESET                  (RESET),
        .H                      (H2),
        .X                      (fir1_y3),
        .X_DV                   (fir1_y3_dv),
        .Y0                     (fir2_y6),
        .Y0_DV                  (fir2_y6_dv),
        .Y1                     (fir2_y7),
        .Y1_DV                  (fir2_y7_dv)
    );

    assign Y0 = fir2_y0;
    assign Y1 = fir2_y1;
    assign Y2 = fir2_y2;
    assign Y3 = fir2_y3;
    assign Y4 = fir2_y4;
    assign Y5 = fir2_y5;
    assign Y6 = fir2_y6;
    assign Y7 = fir2_y7;
    assign Y0_DV = fir2_y0_dv;
    assign Y1_DV = fir2_y1_dv;
    assign Y2_DV = fir2_y2_dv;
    assign Y3_DV = fir2_y3_dv;
    assign Y4_DV = fir2_y4_dv;
    assign Y5_DV = fir2_y5_dv;
    assign Y6_DV = fir2_y6_dv;
    assign Y7_DV = fir2_y7_dv;

endmodule
