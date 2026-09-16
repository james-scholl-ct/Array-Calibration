`include "bcuda_defines.v"
`timescale 1 ns / 1 ps

module bcuda_interp_fir #(
    parameter integer N_TAPS = 15,
    parameter integer N_THREADS = 1,
    parameter integer H_N_BITS = 16,        // M:       s0.(M-1)
    parameter integer X_N_BITS = 16,        // N:       s0.(N-1)
    parameter integer Y_SLICE_N_BITS = 31,  // L+N+M-1: sL.(N+M-2)
    parameter integer Y_N_BITS = 16         // Q:       s0.(Q-1)
)
(
    input   wire                                        CLK,
    input   wire                                        RESET,
    input   wire        [H_N_BITS*(N_TAPS+5)/4 - 1 : 0] H,
    input   wire signed [X_N_BITS - 1 : 0]              X,
    input   wire                                        X_DV,

    output  reg  signed [Y_N_BITS - 1 : 0]              Y0,
    output  reg                                         Y0_DV,
    output  reg  signed [Y_N_BITS - 1 : 0]              Y1,
    output  reg                                         Y1_DV
);


    // SIGNAL       SIZE        FORMAT      COMMENT
    // ------------------------------------------------------------------------
    // H            M           s0.(M-1)
    // X            N           s0.(N-1)
    // slice out    L+N+M-1     sL.(N+M-2)  growth by L due to L1 norm
    // clip out     N+M-1       s0.(N+M-2)  toss L bits
    // Y (round)    Q           s0.(Q-1)    toss (N+M-1) - Q bits
    localparam integer Y_SLICE_INT_BITS = Y_SLICE_N_BITS - X_N_BITS - H_N_BITS + 1;
    localparam integer Y_CLIP_N_BITS = Y_SLICE_N_BITS - Y_SLICE_INT_BITS;

    localparam integer N_SLICES_TOTAL = (N_TAPS + 5) / 4;
    localparam integer N_SLICES = N_SLICES_TOTAL - 1; // subtract center slice

    localparam integer N_DLY_IN = N_THREADS * (2 * N_SLICES - 1) + N_SLICES;
    localparam integer N_DLY_OUT = (N_THREADS + 1) / 2;

    // Four additional samples of latency come from:
    //  1. pipe out of DSP
    //  2. clip pipe
    //  3. round pipe
    //  4. final output pipe
    localparam integer LATENCY_Y0 = (N_SLICES + 2) + 4;
    localparam integer LATENCY_Y1 = LATENCY_Y0 + N_DLY_OUT;
    localparam integer LATENCY_CNT_N_BITS = $clog2(LATENCY_Y1 + 1);
    localparam [LATENCY_CNT_N_BITS - 1 : 0] LATENCY_CNT_TC_Y0 = LATENCY_Y0;
    localparam [LATENCY_CNT_N_BITS - 1 : 0] LATENCY_CNT_TC_Y1 = LATENCY_Y1;

    integer i;
    genvar i_gv;


    // Input delay line
    reg  signed [X_N_BITS - 1 : 0] x_dly [N_DLY_IN - 1 : 0];

    always @(posedge CLK)
    begin
        if (RESET)
        begin
            for (i = 0; i < N_DLY_IN; i = i + 1)
            begin
                x_dly[i] <= 0;
            end
        end
        else if (X_DV)
        begin
            x_dly[0] <= X;
            for (i = 1; i < N_DLY_IN; i = i + 1)
            begin
                x_dly[i] <= x_dly[i - 1];
            end
        end
    end


    // Coefficients: last index is for center tap
    wire signed [H_N_BITS - 1 : 0] h_array [N_SLICES_TOTAL - 1 : 0];

    generate
        for (i_gv = 0; i_gv < N_SLICES_TOTAL; i_gv = i_gv + 1)
        begin : unpack
            assign h_array[i_gv] = $signed(H[i_gv * H_N_BITS +: H_N_BITS]);
        end
    endgenerate


    // DSP slices
    wire signed [X_N_BITS - 1 : 0]          x_tap [2 * N_SLICES - 1 : 0];
    wire signed [Y_SLICE_N_BITS - 1 : 0]    x_acc [N_SLICES - 1 : 0];
    wire signed [Y_SLICE_N_BITS - 1 : 0]    y_acc [N_SLICES - 1 : 0];

    wire signed [X_N_BITS - 1 : 0]          x_tap_center;
    wire signed [Y_SLICE_N_BITS - 1 : 0]    x_acc_center;
    wire signed [Y_SLICE_N_BITS - 1 : 0]    y_acc_center;

    generate
        // delay line taps
        for (i_gv = 0; i_gv < N_SLICES; i_gv = i_gv + 1)
        begin : gen_taps
            assign x_tap[i_gv] =
                    x_dly[N_THREADS * i_gv + N_SLICES - 1 - i_gv];

            assign x_tap[2 * N_SLICES - 1 - i_gv] =
                    x_dly[N_THREADS * (2 * N_SLICES - 1 - i_gv) + N_SLICES - 1 - i_gv];
        end

        assign x_tap_center = x_dly[(N_THREADS + 1) * (N_SLICES - 1)];


        // accumulation cascade
        for (i_gv = 0; i_gv < N_SLICES - 1; i_gv = i_gv + 1)
        begin : gen_acc
            assign x_acc[i_gv] = y_acc[i_gv + 1];
        end

        assign x_acc[N_SLICES - 1] = 0;

        assign x_acc_center = 0;


        // slices
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
                .X_TAP0                 (x_tap[i_gv]),
                .X_TAP1                 (x_tap[2 * N_SLICES - 1 - i_gv]),
                .X_ACC                  (x_acc[i_gv]),
                .Y                      (y_acc[i_gv])
            );
        end

        bcuda_interp_dsp #(
            .X_N_BITS               (X_N_BITS),
            .H_N_BITS               (H_N_BITS),
            .Y_N_BITS               (Y_SLICE_N_BITS))
        i_slice_center (
            .CLK                    (CLK),
            .RESET                  (RESET),
            .H                      (h_array[N_SLICES]),
            .X_DV                   (X_DV),
            .X_TAP0                 (x_tap_center),
            .X_TAP1                 (0),
            .X_ACC                  (x_acc_center),
            .Y                      (y_acc_center)
        );
    endgenerate


    // Pipeline when going back into fabric
    reg  signed [Y_SLICE_N_BITS - 1 : 0]    y_acc_pp;
    reg  signed [Y_SLICE_N_BITS - 1 : 0]    y_acc_center_pp;

    always @(posedge CLK)
    begin
        if (RESET)
        begin
            y_acc_pp <= 0;
            y_acc_center_pp <= 0;
        end
        else if (X_DV)
        begin
            y_acc_pp <= y_acc[0];
            y_acc_center_pp <= y_acc_center;
        end
    end


    // Clip
    wire signed [Y_CLIP_N_BITS - 1 : 0]  y_clip;
    wire signed [Y_CLIP_N_BITS - 1 : 0]  y_clip_center;


    bcuda_clip #(
        .X_N_BITS               (Y_SLICE_N_BITS),
        .Y_N_BITS               (Y_CLIP_N_BITS))
    i_clip (
        .CLK                    (CLK),
        .RESET                  (RESET),
        .X                      (y_acc_pp),
        .X_DV                   (X_DV),
        .Y                      (y_clip)
    );

    bcuda_clip #(
        .X_N_BITS               (Y_SLICE_N_BITS),
        .Y_N_BITS               (Y_CLIP_N_BITS))
    i_clip_center (
        .CLK                    (CLK),
        .RESET                  (RESET),
        .X                      (y_acc_center_pp),
        .X_DV                   (X_DV),
        .Y                      (y_clip_center)
    );


    // Round
    wire signed [Y_N_BITS - 1 : 0]  y_round;
    wire signed [Y_N_BITS - 1 : 0]  y_round_center;

    bcuda_round #(
        .X_N_BITS               (Y_CLIP_N_BITS),
        .Y_N_BITS               (Y_N_BITS))
    i_round (
        .CLK                    (CLK),
        .RESET                  (RESET),
        .X                      (y_clip),
        .X_DV                   (X_DV),
        .Y                      (y_round)
    );

    bcuda_round #(
        .X_N_BITS               (Y_CLIP_N_BITS),
        .Y_N_BITS               (Y_N_BITS))
    i_round_center (
        .CLK                    (CLK),
        .RESET                  (RESET),
        .X                      (y_clip_center),
        .X_DV                   (X_DV),
        .Y                      (y_round_center)
    );


    // Output delay line
    reg  signed [Y_N_BITS - 1 : 0] y_dly [N_DLY_OUT - 1 : 0];

    always @(posedge CLK)
    begin
        if (RESET)
        begin
            for (i = 0; i < N_DLY_OUT; i = i + 1)
            begin
                y_dly[i] <= 0;
            end
        end
        else if (X_DV)
        begin
            y_dly[0] <= y_round_center;
            for (i = 1; i < N_DLY_OUT; i = i + 1)
            begin
                y_dly[i] <= y_dly[i - 1];
            end
        end
    end


    // Mux control
    reg  [LATENCY_CNT_N_BITS - 1 : 0]   latency_cnt;
    wire                                latency_done_y0;
    wire                                latency_done_y1;
    reg  [$clog2(N_THREADS) - 1 : 0]    y_phase_cnt;

    assign latency_done_y0 = (latency_cnt >= LATENCY_CNT_TC_Y0);
    assign latency_done_y1 = (latency_cnt == LATENCY_CNT_TC_Y1);

    always @(posedge CLK)
    begin
        if (RESET)
        begin
            latency_cnt <= 'd0;
            y_phase_cnt <= 'd0;
        end
        else if (X_DV)
        begin
            latency_cnt <= latency_done_y1 ? latency_cnt : latency_cnt + 'd1;
            y_phase_cnt <= latency_done_y0 ? y_phase_cnt + 'd1 : 'd0;
        end
    end


    // Outputs
    always @(posedge CLK)
    begin
        if (RESET)
        begin
            Y0    <= 0;
            Y0_DV <= 1'b0;
            Y1    <= 0;
            Y1_DV <= 1'b0;
        end
        else
        begin
            Y0_DV <= X_DV && latency_done_y0;

            if (X_DV && latency_done_y0)
            begin
                if (y_phase_cnt[$clog2(N_THREADS) - 1])
                begin
                    Y0 <= y_dly[N_DLY_OUT - 1];
                end
                else
                begin
                    Y0 <= y_round;
                end
            end

            Y1_DV <= X_DV && latency_done_y1;

            if (X_DV && latency_done_y1)
            begin
                if (y_phase_cnt[$clog2(N_THREADS) - 1])
                begin
                    Y1 <= y_round;
                end
                else
                begin
                    Y1 <= y_dly[N_DLY_OUT - 1];
                end
            end
        end
    end

endmodule
