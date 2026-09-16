`include "genesis_defines.v"
`timescale 1 ns / 1 ps

module genesis_fifo #(
    parameter integer WIDTH = 32,
    parameter integer DEPTH = 64
)
(
    input   wire                                        S_AXI_ACLK,
    input   wire                                        S_AXI_ARESETN,

    // Read port
    input   wire                                        REN,
    output  wire    [WIDTH - 1 : 0]                     RDATA,
    output  wire                                        EMPTY,
    output  wire    [$clog2(DEPTH) - 1 : 0]             COUNT,

    // Write port
    input   wire                                        WEN,
    input   wire    [WIDTH - 1 : 0]                     WDATA,
    output  wire                                        FULL
);


    localparam integer PTR_N_BITS = $clog2(DEPTH);
    localparam [PTR_N_BITS : 0] FIFO_DEPTH = DEPTH;  // unsigned and sized

    // Pointers
    reg  [PTR_N_BITS - 1 : 0]   wr_ptr;
    wire [PTR_N_BITS     : 0]   wr_ptr_inc;
    wire [PTR_N_BITS - 1 : 0]   wr_ptr_inc_mod;
    reg  [PTR_N_BITS - 1 : 0]   rd_ptr;
    wire [PTR_N_BITS     : 0]   rd_ptr_inc;
    wire [PTR_N_BITS - 1 : 0]   rd_ptr_inc_mod;
    wire [PTR_N_BITS     : 0]   ptr_delta;

    assign wr_ptr_inc = wr_ptr + 'd1;

    assign wr_ptr_inc_mod = (wr_ptr_inc == FIFO_DEPTH) ? 'd0 : wr_ptr_inc;

    assign rd_ptr_inc = rd_ptr + 'd1;

    assign rd_ptr_inc_mod = (rd_ptr_inc == FIFO_DEPTH) ? 'd0 : rd_ptr_inc;

    // COUNT = (wr_ptr - rd_ptr)(mod FIFO_DEPTH) =
    //      [ wr_ptr - rd_ptr              , wr_ptr - rd_ptr >= 0 ]
    //      [ wr_ptr - rd_ptr + FIFO_DEPTH , wr_ptr - rd_ptr <  0 ]
    // wr_ptr and rd_ptr are zero-extended prior to computation
    assign ptr_delta = wr_ptr - rd_ptr;

    assign COUNT = ptr_delta + (ptr_delta[PTR_N_BITS] ? FIFO_DEPTH : 'd0);

    assign FULL = (COUNT == FIFO_DEPTH - 'd1);

    assign EMPTY = (COUNT == 'd0);

    always @(posedge S_AXI_ACLK)
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            wr_ptr <= 'd0;
            rd_ptr <= 'd0;
        end
        else
        begin
            wr_ptr <= WEN && !FULL  ? wr_ptr_inc_mod : wr_ptr;
            rd_ptr <= REN && !EMPTY ? rd_ptr_inc_mod : rd_ptr;
        end
    end


    // Buffer
    reg [WIDTH - 1 : 0] fifo [DEPTH - 1 : 0];

    integer i;

    always @(posedge S_AXI_ACLK)
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            for( i = 0; i < DEPTH; i = i + 1 )
            begin
                fifo[i] <= 'd0;
            end
        end
        else
        begin
            // samples are lost after fifo is filled
            if( WEN && !FULL )
            begin
                fifo[wr_ptr] <= WDATA;
            end
        end
    end

    // RDATA is valid as long as EMPTY is low
    assign RDATA = fifo[rd_ptr];

endmodule
