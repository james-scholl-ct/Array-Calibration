
`timescale 1 ns / 1 ps

module orchid_sync_a2b #(
    parameter integer N_DATA_BITS = 32
)
(
    input                                   A_CLK,
    input       [ N_DATA_BITS - 1 : 0 ]     A_DATA,
    input                                   A_REQ,
    input                                   A_SRESET,
    input                                   B_CLK,

    output reg  [ N_DATA_BITS - 1 : 0 ]     B_DATA,
    output                                  B_DATA_VALID
);


    reg  a2b_req;
    wire b2a_ack;
    reg  b2a_ack_sync0;
    reg  b2a_ack_sync1;

    always @( posedge A_CLK )
    begin
        if ( A_SRESET )
        begin
            a2b_req       <= 1'b0;
            b2a_ack_sync0 <= 1'b0;
            b2a_ack_sync1 <= 1'b0;
        end
        else
        begin
            a2b_req       <= ( A_REQ || a2b_req ) && !b2a_ack_sync1;
            b2a_ack_sync0 <= b2a_ack;
            b2a_ack_sync1 <= b2a_ack_sync0;
        end
    end


    reg                             a2b_req_sync0;
    reg                             a2b_req_sync1;
    reg                             a2b_req_sync2;
    wire                            sample_en;
    reg                             sample_en_dly;

    assign b2a_ack = a2b_req_sync1;

    assign sample_en = a2b_req_sync1 && !a2b_req_sync2;

    always @( posedge B_CLK )
    begin
        a2b_req_sync0 <= a2b_req;
        a2b_req_sync1 <= a2b_req_sync0;
        a2b_req_sync2 <= a2b_req_sync1;
        sample_en_dly <= sample_en;
        if( sample_en )
            B_DATA <= A_DATA;
    end

    assign B_DATA_VALID = sample_en_dly;

endmodule
