`include "bcuda_defines.v"
`timescale 1 ns / 1 ps


module bcuda_clip #(
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

    // Mismatching MSBs imply that the input's sign bits have meaning (i.e.
    // are not merely sign-extended from the would-be output).
    //  +===================+=============+===============+
    //  |      X (s2.1)     |   Y (s0.1)  | N_TOSS+1 MSBs |
    //  |                   |             |   mismatch?   |
    //  +===================+=============+===============+
    //  |    3.5     011.1  |   0.5   0.1 |       1       |
    //  |    3.0     011.0  |   0.5   0.1 |       1       |
    //  |    2.5     010.1  |   0.5   0.1 |       1       |
    //  |    2.0     010.0  |   0.5   0.1 |       1       |
    //  |    1.5     001.1  |   0.5   0.1 |       1       |
    //  |    1.0     001.0  |   0.5   0.1 |       1       |
    //  +-------------------+-------------+---------------+
    //  |    0.5     000.1  |   0.5   0.1 |       0       |
    //  |    0.0     000.0  |   0.0   0.0 |       0       |
    //  |   -0.5     111.1  |  -0.5   1.1 |       0       |
    //  |   -1.0     111.0  |  -1.0   1.0 |       0       |
    //  +-------------------+-------------+---------------+
    //  |   -1.5     110.1  |  -1.0   1.0 |       1       |
    //  |   -2.0     110.0  |  -1.0   1.0 |       1       |
    //  |   -2.5     101.1  |  -1.0   1.0 |       1       |
    //  |   -3.0     101.0  |  -1.0   1.0 |       1       |
    //  |   -3.5     100.1  |  -1.0   1.0 |       1       |
    //  |   -4.0     100.0  |  -1.0   1.0 |       1       |
    //  +===================+=============+===============+

    wire        [N_TOSS : 0]       x_msbs;
    wire                           x_msbs_mismatch;
    wire signed [Y_N_BITS - 1 : 0] y_max;
    wire signed [Y_N_BITS - 1 : 0] y_min;

    assign x_msbs = X[X_N_BITS - 1 : X_N_BITS - 1 - N_TOSS];

    assign x_msbs_mismatch = x_msbs != {(N_TOSS + 1){1'b0}} &&
                             x_msbs != {(N_TOSS + 1){1'b1}};

    assign y_max = $signed({1'b0, {(Y_N_BITS - 1){1'b1}}});
    assign y_min = $signed({1'b1, {(Y_N_BITS - 1){1'b0}}});

    always @(posedge CLK)
    begin
        if (RESET)
        begin
            Y <= 0;
        end
        else if (X_DV)
        begin
            Y <= (x_msbs_mismatch && X[X_N_BITS - 1]) ? y_min :
                 (x_msbs_mismatch                   ) ? y_max :
                 $signed(X[Y_N_BITS - 1 : 0]);
        end
    end

endmodule
