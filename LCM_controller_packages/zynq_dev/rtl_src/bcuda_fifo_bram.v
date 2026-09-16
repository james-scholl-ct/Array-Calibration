`include "bcuda_defines.v"
`timescale 1 ns / 1 ps

module bcuda_fifo_bram #(
    parameter integer WIDTH = 32,
    parameter integer DEPTH = 512
)
(
    input   wire                                        CLK,
    input   wire                                        RESET,

    // Read port
    input   wire                                        REN,
    output  reg                                         RVALID,
    output  reg     [WIDTH - 1 : 0]                     RDATA,
    output  wire    [$clog2(DEPTH) - 1 : 0]             RCOUNT,
    output  wire                                        RERR,
    output  wire                                        ALMOST_EMPTY,
    output  wire                                        EMPTY,

    // Write port
    input   wire                                        WEN,
    input   wire    [WIDTH - 1 : 0]                     WDATA,
    output  wire    [$clog2(DEPTH) - 1 : 0]             WCOUNT,
    output  wire                                        WERR,
    output  wire                                        ALMOST_FULL,
    output  wire                                        FULL
);


    wire [WIDTH - 1 : 0]    rdata_int;
    reg                     ren_dly;


    /////////////////////////////////////////////////////////////////
    // DATA_WIDTH | FIFO_SIZE | FIFO Depth | RDCOUNT/WRCOUNT Width //
    // ===========|===========|============|=======================//
    //   37-72    |  "36Kb"   |     512    |         9-bit         //
    //   19-36    |  "36Kb"   |    1024    |        10-bit         //
    //   19-36    |  "18Kb"   |     512    |         9-bit         // <--
    //   10-18    |  "36Kb"   |    2048    |        11-bit         //
    //   10-18    |  "18Kb"   |    1024    |        10-bit         //
    //    5-9     |  "36Kb"   |    4096    |        12-bit         //
    //    5-9     |  "18Kb"   |    2048    |        11-bit         //
    //    1-4     |  "36Kb"   |    8192    |        13-bit         //
    //    1-4     |  "18Kb"   |    4096    |        12-bit         //
    /////////////////////////////////////////////////////////////////

    FIFO_SYNC_MACRO #(
      .DEVICE               ("7SERIES"),
      .ALMOST_EMPTY_OFFSET  (9'h080),
      .ALMOST_FULL_OFFSET   (9'h080),
      .DATA_WIDTH           (WIDTH),
      .DO_REG               (1'b0),
      .FIFO_SIZE            ("18Kb"))
    i_FIFO_SYNC_MACRO (
      .ALMOSTEMPTY          (ALMOST_EMPTY),
      .ALMOSTFULL           (ALMOST_FULL),
      .CLK                  (CLK),
      .DI                   (WDATA),
      .DO                   (rdata_int),
      .EMPTY                (EMPTY),
      .FULL                 (FULL),
      .RDCOUNT              (RCOUNT),
      .RDEN                 (REN),
      .RDERR                (RERR),
      .RST                  (RESET),
      .WRCOUNT              (WCOUNT),
      .WREN                 (WEN),
      .WRERR                (WERR)
    );

    wire read_valid = ren_dly && !RERR;

    always @(posedge CLK)
    begin
        if (RESET)
        begin
            ren_dly <= 1'b0;
            RVALID  <= 1'b0;
            RDATA   <= 'd0;
        end
        else
        begin
            ren_dly <= REN;
            RVALID  <= read_valid;
            if (read_valid)
                RDATA <= rdata_int;
        end
    end

endmodule
