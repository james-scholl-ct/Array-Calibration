`include "lotus_defines.v"
`timescale 1 ns / 1 ps

module lotus_sync_a2b #(
    parameter integer N_DATA_BITS = 32,
    parameter integer N_DATA2_BITS = 32
)
(
    input   wire                                A_CLK,
    input   wire    [N_DATA_BITS - 1 : 0]       A_DATA,
    input   wire    [N_DATA2_BITS - 1 : 0]      A_DATA2,
    input   wire                                A_REQ,
    input   wire                                A_SRESET,
    input   wire                                B_CLK,

    output  wire                                A_BUSY,
    output  reg     [N_DATA_BITS - 1 : 0]       B_DATA,
    output  reg     [N_DATA2_BITS - 1 : 0]      B_DATA2,
    output  wire                                B_DATA_VALID
);


    localparam integer CONTROL_APPLY_IDX = 3;

    // crossing signals, in addition to A_DATA*
    reg  ax_req;
    wire bx_ack;

    // ------------------------------------------------------------------------
    // A domain
    // ------------------------------------------------------------------------
    reg  a_ack_sync0;
    reg  a_ack_sync1;

    always @( posedge A_CLK )
    begin
        if ( A_SRESET )
        begin
            a_ack_sync0 <= 1'b0;
            a_ack_sync1 <= 1'b0;
            ax_req      <= 1'b0;
        end
        else
        begin
            a_ack_sync0 <= bx_ack;
            a_ack_sync1 <= a_ack_sync0;
            ax_req      <= (A_REQ || ax_req) && !a_ack_sync1;
        end
    end

    // prevent future requests if handshake is currently, or will be, in flight
    assign A_BUSY = (ax_req || a_ack_sync1) ? 1'b1 : A_REQ;

    // ------------------------------------------------------------------------
    // B domain
    // ------------------------------------------------------------------------
    reg  b_req_sync0;
    reg  b_req_sync1;
    reg  b_req_sync2;
    wire sample_en;
    reg  sample_valid;

    assign sample_en = b_req_sync1 && !b_req_sync2;

    always @( posedge B_CLK )
    begin
        b_req_sync0  <= ax_req;
        b_req_sync1  <= b_req_sync0;
        b_req_sync2  <= b_req_sync1;

        sample_valid <= sample_en;

        if( sample_en )
            B_DATA <= A_DATA;

        if( sample_valid && B_DATA[CONTROL_APPLY_IDX] )
            B_DATA2 <= A_DATA2;
    end

    assign bx_ack = b_req_sync2;

    assign B_DATA_VALID = sample_valid;

endmodule
