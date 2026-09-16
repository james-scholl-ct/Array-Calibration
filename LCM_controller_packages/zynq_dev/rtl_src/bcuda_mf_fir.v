`include "bcuda_defines.v"
`timescale 1 ns / 1 ps

// A transposed FIR with non-symmetric coefficients and no TDM. Will
// add TDM later if possible, which can cut DSP slice usage in half.
// The output of the filter is rounded using round-to-even. See the
// fixed-point formatting below.

module bcuda_mf_fir #(
    parameter integer N_TAPS = 16,
    parameter integer H_N_BITS = 16,        // M:     s0.(M-1)
    parameter integer X_N_BITS = 16,        // N:     s0.(N-1)
    parameter integer Y_SLICE_N_BITS = 31,  // N+M-1: s0.(N+M-2)
    parameter integer Y_N_BITS = 16         // Q:     s0.(Q-1)
)
(
    input   wire                                        CLK,
    input   wire                                        RESET,
    input   wire        [H_N_BITS * N_TAPS - 1 : 0]     H,
    input   wire signed [X_N_BITS - 1 : 0]              X,
    input   wire                                        X_DV,

    output  reg  signed [Y_N_BITS - 1 : 0]              Y,
    output  reg                                         Y_DV
);

    localparam integer N_SLICES = N_TAPS;
    localparam integer PIPELINE_IN_SMPLS = 2;
    localparam integer LATENCY_DSP = 2;
    localparam integer LATENCY = PIPELINE_IN_SMPLS + LATENCY_DSP + 3;
    localparam integer LATENCY_CNT_N_BITS = $clog2(LATENCY + 1);
    localparam [LATENCY_CNT_N_BITS - 1 : 0] LATENCY_CNT_TC = LATENCY;

    integer i;
    genvar i_gv;

    // Input delay line
    reg  signed [X_N_BITS - 1 : 0] x_dly [PIPELINE_IN_SMPLS - 1 : 0];

    always @(posedge CLK)
    begin
        if (RESET)
        begin
            for (i = 0; i < PIPELINE_IN_SMPLS; i = i + 1)
            begin
                x_dly[i] <= 0;
            end
        end
        else if (X_DV)
        begin
            x_dly[0] <= X;
            for (i = 1; i < PIPELINE_IN_SMPLS; i = i + 1)
            begin
                x_dly[i] <= x_dly[i - 1];
            end
        end
    end


    // Coefficients
    wire signed [H_N_BITS - 1 : 0] h_array [N_SLICES - 1 : 0];

    generate
        for (i_gv = 0; i_gv < N_SLICES; i_gv = i_gv + 1)
        begin : unpack
            assign h_array[i_gv] = $signed(H[i_gv * H_N_BITS +: H_N_BITS]);
        end
    endgenerate


    // DSP slices
    wire signed [Y_SLICE_N_BITS - 1 : 0]    x_acc [N_SLICES - 1 : 0];
    wire signed [Y_SLICE_N_BITS - 1 : 0]    y_acc [N_SLICES - 1 : 0];

    generate
        //  - accumulation cascade
        assign x_acc[N_SLICES - 1] = 0;

        for (i_gv = 0; i_gv < N_SLICES - 1; i_gv = i_gv + 1)
        begin : gen_acc
            assign x_acc[i_gv] = y_acc[i_gv + 1];
        end

        // - slices
        for (i_gv = 0; i_gv < N_SLICES; i_gv = i_gv + 1)
        begin : gen_slices
            bcuda_interp_dsp #(
                .X_N_BITS               (X_N_BITS),
                .H_N_BITS               (H_N_BITS),
                .Y_N_BITS               (Y_SLICE_N_BITS))
            i_slice (
                .CLK                    (CLK),
                .RESET                  (RESET),
                .H                      (h_array[i_gv]),
                .X_DV                   (X_DV),
                .X_TAP0                 (x_dly[PIPELINE_IN_SMPLS - 1]),
                .X_TAP1                 (0),
                .X_ACC                  (x_acc[i_gv]),
                .Y                      (y_acc[i_gv])
            );
        end
    endgenerate


    // Pipeline when going back into fabric
    reg  signed [Y_SLICE_N_BITS - 1 : 0]    y_acc_pp;

    always @(posedge CLK)
    begin
        if (RESET)
        begin
            y_acc_pp <= 0;
        end
        else if (X_DV)
        begin
            y_acc_pp <= y_acc[0];
        end
    end


    // Round
    wire signed [Y_N_BITS - 1 : 0]  y_round;

    bcuda_round #(
        .X_N_BITS               (Y_SLICE_N_BITS),
        .Y_N_BITS               (Y_N_BITS))
    i_round (
        .CLK                    (CLK),
        .RESET                  (RESET),
        .X                      (y_acc_pp),
        .X_DV                   (X_DV),
        .Y                      (y_round)
    );


    // Y_DV control
    // - count samples and release the output at the right time
    reg  [LATENCY_CNT_N_BITS - 1 : 0]   latency_cnt;
    wire                                latency_done;

    assign latency_done = (latency_cnt == LATENCY_CNT_TC);

    always @(posedge CLK)
    begin
        if (RESET)
        begin
            latency_cnt <= 'd0;
        end
        else if (X_DV && !latency_done)
        begin
            latency_cnt <= latency_cnt + 'd1;
        end
    end


    // Outputs
    always @(posedge CLK)
    begin
        if (RESET)
        begin
            Y    <= 0;
            Y_DV <= 1'b0;
        end
        else
        begin
            Y_DV <= X_DV && latency_done;

            if (X_DV && latency_done)
            begin
                Y <= y_round;
            end
        end
    end

endmodule
