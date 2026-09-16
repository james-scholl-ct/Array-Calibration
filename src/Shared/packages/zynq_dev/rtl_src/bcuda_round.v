`include "bcuda_defines.v"
`timescale 1 ns / 1 ps

// Convergent rounding to even to match numpy.

module bcuda_round #(
    parameter integer X_N_BITS = 32,
    parameter integer Y_N_BITS = 16
)
(
    input   wire                                        CLK,
    input   wire                                        RESET,
    input   wire signed [X_N_BITS - 1 : 0]              X,
    input   wire                                        X_DV,

    output  reg  signed [Y_N_BITS - 1 : 0]              Y
);


    localparam integer N_TOSS = X_N_BITS - Y_N_BITS;

    //  +===================+=======+============+=======+=======+
    //  |      X (s1.2)     |  sum  |  Y (s1.0)  | half? | clip? |
    //  +===================+=======+============+=======+=======+
    //  |    1.75    01.11  | 10.01 |   1    01. |   0   |   1   |
    //  |    1.50    01.10  | 10.00 |   1    01. |   1   |   1   |
    //  |    1.25    01.01  | 01.11 |   1    01. |   0   |   0   |
    //  |    1.00    01.00  | 01.10 |   1    01. |   0   |   0   |
    //  |    0.75    00.11  | 01.01 |   1    01. |   0   |   0   |
    //  |    0.50    00.10  | 01.00 |   0    00. |   1   |   0   |
    //  |    0.25    00.01  | 00.11 |   0    00. |   0   |   0   |
    //  |    0.00    00.00  | 00.10 |   0    00. |   0   |   0   |
    //  |   -0.25    11.11  | 00.01 |   0    00. |   0   |   0   |
    //  |   -0.50    11.10  | 00.00 |   0    00. |   1   |   0   |
    //  |   -0.75    11.01  | 11.11 |  -1    11. |   0   |   0   |
    //  |   -1.00    11.00  | 11.10 |  -1    11. |   0   |   0   |
    //  |   -1.25    10.11  | 11.01 |  -1    11. |   0   |   0   |
    //  |   -1.50    10.10  | 11.00 |  -2    10. |   1   |   0   |
    //  |   -1.75    10.01  | 10.11 |  -2    10. |   0   |   0   |
    //  |   -2.00    10.00  | 10.10 |  -2    10. |   0   |   0   |
    //  +===================+=======+============+=======+=======+

    wire signed [X_N_BITS - 1 : 0]  half;
    wire signed [X_N_BITS - 1 : 0]  sum;
    wire signed [N_TOSS - 1 : 0]    pat_half;
    wire signed [Y_N_BITS : 0]      pat_clip;
    wire                            detect_half;
    wire                            detect_clip;
    wire signed [Y_N_BITS - 1 : 0]  y_max;

    assign half = 1 << (N_TOSS - 1);

    assign sum = X + half;

    assign pat_half = {1'b1, {(N_TOSS - 1){1'b0}}};
    assign pat_clip = {1'b0, {Y_N_BITS{1'b1}}};

    assign detect_half = (X[N_TOSS - 1 : 0] == pat_half);
    assign detect_clip = (X[X_N_BITS - 1 : N_TOSS - 1] == pat_clip);

    assign y_max = {1'b0, {(Y_N_BITS - 1){1'b1}}};

    always @(posedge CLK)
    begin
        if (RESET)
        begin
            Y <= 0;
        end
        else if (X_DV)
        begin
            Y <= detect_clip ? y_max :
                 detect_half ? {sum[X_N_BITS - 1 : N_TOSS + 1], 1'b0} :
                               sum[X_N_BITS - 1 : N_TOSS];
        end
    end

endmodule
