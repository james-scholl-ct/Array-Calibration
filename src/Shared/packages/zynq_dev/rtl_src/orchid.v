
`timescale 1 ns / 1 ps

module orchid #(
    parameter integer C_S00_AXI_DATA_DEPTH = 72,
    parameter integer C_S00_AXI_DATA_WIDTH = 32,
    parameter integer C_S00_AXI_ADDR_WIDTH = 9
)
(
    // AXI interface connections
    //      Global signals
    input   wire                                        s00_axi_aclk,
    input   wire                                        s00_axi_aresetn,
    //      (AW) Write Address channel
    input   wire    [C_S00_AXI_ADDR_WIDTH-1 : 0]        s00_axi_awaddr,
    input   wire    [2 : 0]                             s00_axi_awprot,
    input   wire                                        s00_axi_awvalid,
    output  wire                                        s00_axi_awready,
    //      (W) Write Data channel
    input   wire    [C_S00_AXI_DATA_WIDTH-1 : 0]        s00_axi_wdata,
    input   wire    [(C_S00_AXI_DATA_WIDTH/8)-1 : 0]    s00_axi_wstrb,
    input   wire                                        s00_axi_wvalid,
    output  wire                                        s00_axi_wready,
    //      (B) Write Response channel
    input   wire                                        s00_axi_bready,
    output  wire    [1 : 0]                             s00_axi_bresp,
    output  wire                                        s00_axi_bvalid,
    //      (AR) Read Address channel
    input   wire    [C_S00_AXI_ADDR_WIDTH-1 : 0]        s00_axi_araddr,
    input   wire    [2 : 0]                             s00_axi_arprot,
    input   wire                                        s00_axi_arvalid,
    output  wire                                        s00_axi_arready,
    //      (R) Read Data channel
    input   wire                                        s00_axi_rready,
    output  wire    [C_S00_AXI_DATA_WIDTH-1 : 0]        s00_axi_rdata,
    output  wire    [1 : 0]                             s00_axi_rresp,
    output  wire                                        s00_axi_rvalid,

    // IP connections
    input   wire                                        CLK_120,

    output  wire                                        LVDS_0N,
    output  wire                                        LVDS_0P,
    output  wire                                        LVDS_1N,
    output  wire                                        LVDS_1P,
    output  wire                                        LVDS_2N,
    output  wire                                        LVDS_2P,
    output  wire                                        LVDS_3N,
    output  wire                                        LVDS_3P,
    output  wire                                        LVDS_4N,
    output  wire                                        LVDS_4P,
    output  wire                                        LVDS_5N,
    output  wire                                        LVDS_5P,
    output  wire                                        LVDS_CLK_N,
    output  wire                                        LVDS_CLK_P,
    output  wire                                        POL,
    output  wire                                        TP1
);


    localparam integer MEMORY_N_BITS = C_S00_AXI_DATA_DEPTH *
                                       C_S00_AXI_DATA_WIDTH;


    wire [ C_S00_AXI_DATA_DEPTH - 1 : 0 ]   axi_write_enables;
    wire [ MEMORY_N_BITS - 1 : 0 ]          axi_memory;


    orchid_axi_interface #(
        .C_S_AXI_DATA_DEPTH     (C_S00_AXI_DATA_DEPTH),
        .C_S_AXI_DATA_WIDTH     (C_S00_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH     (C_S00_AXI_ADDR_WIDTH))
    I_orchid_axi_interface (
        // AXI interface connections
        .S_AXI_ACLK             (s00_axi_aclk),
        .S_AXI_ARESETN          (s00_axi_aresetn),

        .S_AXI_AWADDR           (s00_axi_awaddr),
        .S_AXI_AWPROT           (s00_axi_awprot),
        .S_AXI_AWVALID          (s00_axi_awvalid),
        .S_AXI_AWREADY          (s00_axi_awready),

        .S_AXI_WDATA            (s00_axi_wdata),
        .S_AXI_WSTRB            (s00_axi_wstrb),
        .S_AXI_WVALID           (s00_axi_wvalid),
        .S_AXI_WREADY           (s00_axi_wready),

        .S_AXI_BREADY           (s00_axi_bready),
        .S_AXI_BRESP            (s00_axi_bresp),
        .S_AXI_BVALID           (s00_axi_bvalid),

        .S_AXI_ARADDR           (s00_axi_araddr),
        .S_AXI_ARPROT           (s00_axi_arprot),
        .S_AXI_ARVALID          (s00_axi_arvalid),
        .S_AXI_ARREADY          (s00_axi_arready),

        .S_AXI_RREADY           (s00_axi_rready),
        .S_AXI_RDATA            (s00_axi_rdata),
        .S_AXI_RRESP            (s00_axi_rresp),
        .S_AXI_RVALID           (s00_axi_rvalid),

        // IP connections
        .AXI_WRITE_ENABLES      (axi_write_enables),
        .AXI_MEMORY             (axi_memory)
    );


    wire [C_S00_AXI_DATA_WIDTH - 1 : 0 ] axi_memory_u [C_S00_AXI_DATA_DEPTH - 1 : 0];

    genvar i_gv;
    generate
        for( i_gv = 0; i_gv < C_S00_AXI_DATA_DEPTH; i_gv = i_gv + 1 )
        begin : array_reshape
            assign axi_memory_u[ i_gv ] =
                axi_memory[ C_S00_AXI_DATA_WIDTH * (i_gv+1) - 1 : C_S00_AXI_DATA_WIDTH * i_gv ];
        end
    endgenerate


    // CDC from master to slave
    wire [C_S00_AXI_DATA_WIDTH-1 : 0]   a_data;
    wire                                a_req;
    wire [C_S00_AXI_DATA_WIDTH-1 : 0]   b_data;
    wire                                b_data_valid;

    assign a_data = axi_memory_u[C_S00_AXI_DATA_DEPTH - 1];

    assign a_req = axi_write_enables[C_S00_AXI_DATA_DEPTH - 1];

    orchid_sync_a2b #(
        .N_DATA_BITS    (C_S00_AXI_DATA_WIDTH))
    I_orchid_sync_a2b (
        .A_CLK          (s00_axi_aclk),
        .A_DATA         (a_data),
        .A_REQ          (a_req),
        .A_SRESET       (~s00_axi_aresetn),
        .B_CLK          (CLK_120),

        .B_DATA         (b_data),
        .B_DATA_VALID   (b_data_valid)
    );


    orchid_core #(
        .C_S_AXI_DATA_DEPTH     (C_S00_AXI_DATA_DEPTH),
        .C_S_AXI_DATA_WIDTH     (C_S00_AXI_DATA_WIDTH))
    I_orchid_core (
        .CLK                    (CLK_120),
        .AXI_MEMORY             (axi_memory),
        .CONTROL                (b_data),
        .CONTROL_VALID          (b_data_valid),

        .LVDS_0N                (LVDS_0N),
        .LVDS_0P                (LVDS_0P),
        .LVDS_1N                (LVDS_1N),
        .LVDS_1P                (LVDS_1P),
        .LVDS_2N                (LVDS_2N),
        .LVDS_2P                (LVDS_2P),
        .LVDS_3N                (LVDS_3N),
        .LVDS_3P                (LVDS_3P),
        .LVDS_4N                (LVDS_4N),
        .LVDS_4P                (LVDS_4P),
        .LVDS_5N                (LVDS_5N),
        .LVDS_5P                (LVDS_5P),
        .LVDS_CLK_N             (LVDS_CLK_N),
        .LVDS_CLK_P             (LVDS_CLK_P),
        .POL                    (POL),
        .TP1                    (TP1)
    );

endmodule
