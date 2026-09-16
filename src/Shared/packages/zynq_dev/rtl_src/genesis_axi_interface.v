`include "genesis_defines.v"
`timescale 1 ns / 1 ps

// AXI4-Lite:
//      * all transactions are of burst length 1
//      * all data accesses use the full width of the data bus
//      * all accesses are Non-modifiable, Non-bufferable
//      * exclusive accesses are not supported

module genesis_axi_interface #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 12
)
(
    // AXI - Global signals
    input   wire                                        S_AXI_ACLK,
    input   wire                                        S_AXI_ARESETN,

    // AXI - (AW) Write Address channel
    input   wire    [C_S_AXI_ADDR_WIDTH - 1 : 0]        S_AXI_AWADDR,
    input   wire    [2 : 0]                             S_AXI_AWPROT,
    input   wire                                        S_AXI_AWVALID,
    output  wire                                        S_AXI_AWREADY,

    // AXI - (W) Write Data channel
    input   wire    [C_S_AXI_DATA_WIDTH - 1 : 0]        S_AXI_WDATA,
    input   wire    [(C_S_AXI_DATA_WIDTH / 8) - 1 : 0]  S_AXI_WSTRB,
    input   wire                                        S_AXI_WVALID,
    output  wire                                        S_AXI_WREADY,

    // AXI - (B) Write Response channel
    input   wire                                        S_AXI_BREADY,
    output  wire    [1 : 0]                             S_AXI_BRESP,
    output  wire                                        S_AXI_BVALID,

    // AXI - (AR) Read Address channel
    input   wire    [C_S_AXI_ADDR_WIDTH - 1 : 0]        S_AXI_ARADDR,
    input   wire    [2 : 0]                             S_AXI_ARPROT,
    input   wire                                        S_AXI_ARVALID,
    output  wire                                        S_AXI_ARREADY,

    // AXI - (R) Read Data channel
    input   wire                                        S_AXI_RREADY,
    output  wire    [C_S_AXI_DATA_WIDTH - 1  : 0]       S_AXI_RDATA,
    output  wire    [1 : 0]                             S_AXI_RRESP,
    output  wire                                        S_AXI_RVALID,

    // Downstream write interface
    input   wire                                        WR_ACK,
    output  wire                                        WR_REQ,
    output  wire    [C_S_AXI_ADDR_WIDTH - 1 : 0]        WR_ADDR,
    output  wire    [C_S_AXI_DATA_WIDTH - 1 : 0]        WR_DATA,
    output  wire    [(C_S_AXI_DATA_WIDTH / 8) - 1 : 0]  WR_STRB,

    // Downstream read interface
    input   wire                                        RD_VALID,
    input   wire    [C_S_AXI_DATA_WIDTH - 1  : 0]       RD_DATA,
    output  wire                                        RD_EN,
    output  wire    [C_S_AXI_ADDR_WIDTH - 1 : 0]        RD_ADDR
);


    // ------------------------------------------------------------------------
    // Write
    // ------------------------------------------------------------------------
    reg  write_busy;

    always @( posedge S_AXI_ACLK )
    begin
        if ( S_AXI_ARESETN == 1'b0 )
        begin
            write_busy    <= 1'b0;
        end
        else
        begin
            if( S_AXI_BREADY && S_AXI_BVALID )
            begin
                write_busy <= 1'b0;
            end
            else if( WR_ACK )
            begin
                write_busy <= 1'b1;
            end
        end
    end

    // axi interface
    assign S_AXI_AWREADY = WR_ACK;
    assign S_AXI_WREADY = WR_ACK;
    assign S_AXI_BRESP = 2'b00;
    assign S_AXI_BVALID = write_busy;

    // memory interface
    assign WR_REQ = S_AXI_AWVALID && S_AXI_WVALID && !WR_ACK && !write_busy;
    assign WR_ADDR = S_AXI_AWADDR;
    assign WR_DATA = S_AXI_WDATA;
    assign WR_STRB = S_AXI_WSTRB;

    // ------------------------------------------------------------------------
    // Read
    // ------------------------------------------------------------------------
    reg                                     read_busy;
    reg                                     read_valid;
    reg     [C_S_AXI_DATA_WIDTH - 1  : 0]   read_data_pp;

    always @( posedge S_AXI_ACLK )
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            read_busy    <= 1'b0;
            read_valid   <= 1'b0;
            read_data_pp <= 'd0;
        end
        else
        begin
            if( S_AXI_RREADY && S_AXI_RVALID )
            begin
                read_busy <= 1'b0;
            end
            else if( S_AXI_ARVALID )
            begin
                read_busy <= 1'b1;
            end

            if( S_AXI_RREADY && S_AXI_RVALID )
            begin
                read_valid <= 1'b0;
            end
            else if( RD_VALID )
            begin
                read_valid <= 1'b1;
            end

            if( RD_VALID )
            begin
                read_data_pp <= RD_DATA;
            end
        end
    end

    // axi interface
    assign S_AXI_ARREADY = !read_busy;
    assign S_AXI_RDATA = read_data_pp;
    assign S_AXI_RVALID = read_valid;
    assign S_AXI_RRESP = 2'b00;

    // memory interface
    assign RD_EN = S_AXI_ARREADY && S_AXI_ARVALID;
    assign RD_ADDR = S_AXI_ARADDR;

endmodule
