`include "bcuda_defines.v"
`timescale 1 ns / 1 ps

module bcuda_ip1 #(
    parameter integer C_S00_AXI_DATA_DEPTH = 64,
    parameter integer C_S00_AXI_DATA_WIDTH = 32,
    parameter integer C_S00_AXI_ADDR_WIDTH = 12,
    parameter integer GPI_N_BITS           = 2,
    parameter integer GPO_N_BITS           = 2
)
(
    // AXI connections
    //      Global signals
    input   wire                                        s00_axi_aclk,
    input   wire                                        s00_axi_aresetn,
    //      (AW) Write Address channel
    input   wire    [C_S00_AXI_ADDR_WIDTH - 1 : 0]      s00_axi_awaddr,
    input   wire    [2 : 0]                             s00_axi_awprot,
    input   wire                                        s00_axi_awvalid,
    output  wire                                        s00_axi_awready,
    //      (W) Write Data channel
    input   wire    [C_S00_AXI_DATA_WIDTH - 1 : 0]      s00_axi_wdata,
    input   wire    [(C_S00_AXI_DATA_WIDTH/8) - 1 : 0]  s00_axi_wstrb,
    input   wire                                        s00_axi_wvalid,
    output  wire                                        s00_axi_wready,
    //      (B) Write Response channel
    input   wire                                        s00_axi_bready,
    output  wire    [1 : 0]                             s00_axi_bresp,
    output  wire                                        s00_axi_bvalid,
    //      (AR) Read Address channel
    input   wire    [C_S00_AXI_ADDR_WIDTH - 1 : 0]      s00_axi_araddr,
    input   wire    [2 : 0]                             s00_axi_arprot,
    input   wire                                        s00_axi_arvalid,
    output  wire                                        s00_axi_arready,
    //      (R) Read Data channel
    input   wire                                        s00_axi_rready,
    output  wire    [C_S00_AXI_DATA_WIDTH - 1 : 0]      s00_axi_rdata,
    output  wire    [1 : 0]                             s00_axi_rresp,
    output  wire                                        s00_axi_rvalid,

    // IP connections
    input   wire    [GPI_N_BITS - 1 : 0]                GPI,
    output  wire    [GPO_N_BITS - 1 : 0]                GPO
);


    // Paramers: Active Memory
    //              +================+=======================+
    //              |  WORD ADDRESS  |  MEMORY               |
    //              +----------------+-----------------------+
    //              |  0*L + 0       |  block   0, word   0  |
    //              |  0*L + 1       |  block   0, word   1  |
    //              |  ...           |  ...                  |
    //              |  0*L + L-2     |  block   0, word L-2  |
    //              |  0*L + L-1     |  block   0, word L-1  |
    //              +----------------+-----------------------+
    //              |  1*L + 0       |  block   1, word   0  |
    //              |  1*L + 1       |  block   1, word   1  |
    //              |  ...           |  ...                  |
    //              |  1*L + L-2     |  block   1, word L-2  |
    //              |  1*L + L-1     |  block   1, word L-1  |
    //              +----------------+-----------------------+
    //              |  ...           |  ...                  |
    //              +----------------+-----------------------+
    //              |  (K-2)*L + 0   |  block K-2, word   0  |
    //              |  (K-2)*L + 1   |  block K-2, word   1  |
    //              |  ...           |  ...                  |
    //              |  (K-2)*L + L-3 |  block K-2, word L-3  |  DEBUG_WORD
    //              |  (K-2)*L + L-2 |  block K-2, word L-2  |  GPO_WORD
    //              |  (K-2)*L + L-1 |  block K-2, word L-1  |  CMD_FIFO_WORD
    //              +----------------+-----------------------+
    // STATUS BLOCK |  (K-1)*L + 0   |  block K-1, word   0  |  RSP_FIFO_WORD
    //              |  (K-1)*L + 1   |  block K-1, word   1  |  RSP_FIFO_STATUS_WORD
    //              |  (K-1)*L + 2   |  block K-1, word   2  |  CMD_FIFO_STATUS_WORD
    //              |  (K-1)*L + 3   |  block K-1, word   3  |  GPI_WORD
    //              |  ...           |  ...                  |
    //              |  (K-1)*L + L-2 |  block K-1, word L-2  |
    //              |  (K-1)*L + L-1 |  block K-1, word L-1  |
    //              +----------------+-----------------------+
    localparam integer MEM_N_BLOCKS = 8;
    localparam integer MEM_N_STATUS_BLOCKS = 1;

    localparam integer MEM_N_BITS = C_S00_AXI_DATA_DEPTH * C_S00_AXI_DATA_WIDTH;
    localparam integer MEM_N_WORDS_PER_BLOCK = C_S00_AXI_DATA_DEPTH / MEM_N_BLOCKS;
    localparam integer MEM_N_STATUS_WORDS = MEM_N_STATUS_BLOCKS * MEM_N_WORDS_PER_BLOCK;
    localparam integer MEM_N_STATUS_BITS = MEM_N_STATUS_WORDS * C_S00_AXI_DATA_WIDTH;
    localparam integer MEM_STATUS_OFFSET = C_S00_AXI_DATA_DEPTH - MEM_N_STATUS_WORDS;

    // Parameters: FIFOs
    localparam integer CMD_FIFO_DEPTH = 512;
    localparam integer CMD_FIFO_COUNT_N_BITS = $clog2(CMD_FIFO_DEPTH);
    localparam integer CMD_FIFO_WIDTH = C_S00_AXI_DATA_WIDTH;

    localparam integer RSP_FIFO_DEPTH = 512;
    localparam integer RSP_FIFO_COUNT_N_BITS = $clog2(RSP_FIFO_DEPTH);
    localparam integer RSP_FIFO_WIDTH = C_S00_AXI_DATA_WIDTH - 1;


    // Downstream write interfaces to active memory
    wire                                            axi_wr_ack;
    wire                                            axi_wr_req;
    wire    [C_S00_AXI_ADDR_WIDTH - 1 : 0]          axi_wr_addr;
    wire    [C_S00_AXI_DATA_WIDTH - 1 : 0]          axi_wr_data;
    wire    [(C_S00_AXI_DATA_WIDTH / 8) - 1 : 0]    axi_wr_strb;

    // Downstream read interfaces to active memory
    wire                                            axi_rd_valid;
    wire    [C_S00_AXI_DATA_WIDTH - 1  : 0]         axi_rd_data;
    wire                                            axi_rd_en;
    wire    [C_S00_AXI_ADDR_WIDTH - 1 : 0]          axi_rd_addr;

    // FIFO interfaces
    wire                                            cmd_fifo_ren;
    wire                                            cmd_fifo_rvalid;
    wire    [CMD_FIFO_WIDTH - 1 : 0]                cmd_fifo_rdata;
    wire    [CMD_FIFO_COUNT_N_BITS - 1 : 0]         cmd_fifo_rcount;
    wire                                            cmd_fifo_rerr;
    wire                                            cmd_fifo_almost_empty;
    wire                                            cmd_fifo_empty;
    reg                                             cmd_fifo_wen;
    wire    [CMD_FIFO_WIDTH - 1 : 0]                cmd_fifo_wdata;
    wire    [CMD_FIFO_COUNT_N_BITS - 1 : 0]         cmd_fifo_wcount;
    wire                                            cmd_fifo_werr;
    wire                                            cmd_fifo_almost_full;
    wire                                            cmd_fifo_full;

    wire                                            rsp_fifo_ren;
    wire                                            rsp_fifo_rvalid;
    wire    [RSP_FIFO_WIDTH - 1 : 0]                rsp_fifo_rdata;
    wire    [RSP_FIFO_COUNT_N_BITS - 1 : 0]         rsp_fifo_rcount;
    wire                                            rsp_fifo_rerr;
    wire                                            rsp_fifo_almost_empty;
    wire                                            rsp_fifo_empty;
    wire                                            rsp_fifo_wen;
    wire    [RSP_FIFO_WIDTH - 1 : 0]                rsp_fifo_wdata;
    wire    [RSP_FIFO_COUNT_N_BITS - 1 : 0]         rsp_fifo_wcount;
    wire                                            rsp_fifo_werr;
    wire                                            rsp_fifo_almost_full;
    wire                                            rsp_fifo_full;


    // ------------------------------------------------------------------------
    // AXI4-Lite controller
    // ------------------------------------------------------------------------
    bcuda_axi_interface #(
        .C_S_AXI_DATA_WIDTH     (C_S00_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH     (C_S00_AXI_ADDR_WIDTH))
    i_axi_interface (
        // Upstream AXI4-Lite interface
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

        // Downstream write interface
        .WR_ACK                 (axi_wr_ack),
        .WR_REQ                 (axi_wr_req),
        .WR_ADDR                (axi_wr_addr),
        .WR_DATA                (axi_wr_data),
        .WR_STRB                (axi_wr_strb),

        // Downstream read interface
        .RD_VALID               (axi_rd_valid),
        .RD_DATA                (axi_rd_data),
        .RD_EN                  (axi_rd_en),
        .RD_ADDR                (axi_rd_addr)
    );


    // ------------------------------------------------------------------------
    // Active Memory
    // ------------------------------------------------------------------------
    wire [MEM_N_BITS - 1 : 0]           active_mem;
    wire [C_S00_AXI_DATA_DEPTH - 1 : 0] active_mem_rd_enables;
    wire [C_S00_AXI_DATA_DEPTH - 1 : 0] active_mem_wr_enables;
    wire [MEM_N_STATUS_BITS - 1 : 0]    active_mem_status;
    wire                                active_mem_wr_back_pressure;

    assign active_mem_wr_back_pressure = 1'b0;

    bcuda_active_mem #(
        .C_S_AXI_DATA_DEPTH     (C_S00_AXI_DATA_DEPTH),
        .C_S_AXI_DATA_WIDTH     (C_S00_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH     (C_S00_AXI_ADDR_WIDTH),
        .N_STATUS_WORDS         (MEM_N_STATUS_WORDS))
    i_active_mem (
        .S_AXI_ACLK             (s00_axi_aclk),
        .S_AXI_ARESETN          (s00_axi_aresetn),

        // Upstream write interfaces
        .CH0_WR_REQ             (axi_wr_req),
        .CH0_WR_ADDR            (axi_wr_addr),
        .CH0_WR_DATA            (axi_wr_data),
        .CH0_WR_STRB            (axi_wr_strb),
        .CH0_WR_ACK             (axi_wr_ack),

        .CH1_WR_REQ             (1'b0),
        .CH1_WR_ADDR            ('d0),
        .CH1_WR_DATA            ('d0),
        .CH1_WR_STRB            ('d0),
        .CH1_WR_ACK             (),

        // Upstream read interfaces
        .CH0_RD_EN              (axi_rd_en),
        .CH0_RD_ADDR            (axi_rd_addr),
        .CH0_RD_VALID           (axi_rd_valid),
        .CH0_RD_DATA            (axi_rd_data),

        // Downstream interface
        .STATUS                 (active_mem_status),
        .WR_BACK_PRESSURE       (active_mem_wr_back_pressure),
        .MEMORY                 (active_mem),
        .READ_ENABLES           (active_mem_rd_enables),
        .WRITE_ENABLES          (active_mem_wr_enables)
    );

    genvar i_gv;
    wire [C_S00_AXI_DATA_WIDTH - 1 : 0] active_mem_u [C_S00_AXI_DATA_DEPTH - 1 : 0];
    wire [C_S00_AXI_DATA_WIDTH - 1 : 0] active_mem_status_u [C_S00_AXI_DATA_DEPTH - 1 : 0];

    // Unpack the memory (R/W and Write-only words)
    generate
        for (i_gv = 0; i_gv < C_S00_AXI_DATA_DEPTH; i_gv = i_gv + 1)
        begin : unpack
            assign active_mem_u[i_gv] =
                active_mem[i_gv * C_S00_AXI_DATA_WIDTH +: C_S00_AXI_DATA_WIDTH];
        end
    endgenerate


    // Pack the status words (read-only words)
    generate
        for (i_gv = 0; i_gv < MEM_N_STATUS_WORDS; i_gv = i_gv + 1)
        begin : pack
            assign active_mem_status[i_gv * C_S00_AXI_DATA_WIDTH +: C_S00_AXI_DATA_WIDTH] =
                active_mem_status_u[i_gv];
        end
    endgenerate


    // ------------------------------------------------------------------------
    // Memory map and miscellaneous control
    // ------------------------------------------------------------------------
    localparam integer DEBUG_WORD           = MEM_STATUS_OFFSET - 3;
    localparam integer GPO_WORD             = MEM_STATUS_OFFSET - 2;
    localparam integer CMD_FIFO_WORD        = MEM_STATUS_OFFSET - 1;
    localparam integer RSP_FIFO_WORD        = MEM_STATUS_OFFSET;
    localparam integer RSP_FIFO_STATUS_WORD = MEM_STATUS_OFFSET + 1;
    localparam integer CMD_FIFO_STATUS_WORD = MEM_STATUS_OFFSET + 2;
    localparam integer GPI_WORD             = MEM_STATUS_OFFSET + 3;


    // Populate status words
    wire [GPI_N_BITS - 1 : 0] gpi_status;

    assign active_mem_status_u[0] = {rsp_fifo_rvalid, rsp_fifo_rdata};
    assign active_mem_status_u[1] = {rsp_fifo_rerr,
                                     rsp_fifo_empty,
                                     rsp_fifo_almost_empty,
                                     rsp_fifo_rcount,
                                     rsp_fifo_werr,
                                     rsp_fifo_full,
                                     rsp_fifo_almost_full,
                                     rsp_fifo_wcount};
    assign active_mem_status_u[2] = {cmd_fifo_rerr,
                                     cmd_fifo_empty,
                                     cmd_fifo_almost_empty,
                                     cmd_fifo_rcount,
                                     cmd_fifo_werr,
                                     cmd_fifo_full,
                                     cmd_fifo_almost_full,
                                     cmd_fifo_wcount};
    assign active_mem_status_u[3] = gpi_status;
    assign active_mem_status_u[4] = 'd0;
    assign active_mem_status_u[5] = 'd0;
    assign active_mem_status_u[6] = 'd0;
    assign active_mem_status_u[7] = 'd0;


    // Config and control
    wire            cmd_fifo_hold;
    wire [3 : 0]    rsp_fifo_sig_sel;

    always @(posedge s00_axi_aclk)
    begin
        if (s00_axi_aresetn == 1'b0)
        begin
            cmd_fifo_wen <= 1'b0;
        end
        else
        begin
            // write the fifo after data has been written to memory
            cmd_fifo_wen <= active_mem_wr_enables[CMD_FIFO_WORD];
        end
    end

    assign cmd_fifo_wdata = active_mem_u[CMD_FIFO_WORD];

    assign rsp_fifo_ren = active_mem_rd_enables[RSP_FIFO_WORD];

    assign cmd_fifo_hold = active_mem_u[DEBUG_WORD][0];
    assign rsp_fifo_sig_sel = active_mem_u[DEBUG_WORD][4 : 1];

    // ------------------------------------------------------------------------
    // FIFOs
    // ------------------------------------------------------------------------
    bcuda_fifo_bram #(
        .DEPTH                  (CMD_FIFO_DEPTH),
        .WIDTH                  (CMD_FIFO_WIDTH))
    i_cmd_fifo (
        .CLK                    (s00_axi_aclk),
        .RESET                  (~s00_axi_aresetn),

        // Read port
        .REN                    (cmd_fifo_ren),
        .RVALID                 (cmd_fifo_rvalid),
        .RDATA                  (cmd_fifo_rdata),
        .RCOUNT                 (cmd_fifo_rcount),
        .RERR                   (cmd_fifo_rerr),
        .ALMOST_EMPTY           (cmd_fifo_almost_empty),
        .EMPTY                  (cmd_fifo_empty),

        // Write port
        .WEN                    (cmd_fifo_wen),
        .WDATA                  (cmd_fifo_wdata),
        .WCOUNT                 (cmd_fifo_wcount),
        .WERR                   (cmd_fifo_werr),
        .ALMOST_FULL            (cmd_fifo_almost_full),
        .FULL                   (cmd_fifo_full)
    );

    bcuda_fifo_bram #(
        .DEPTH                  (RSP_FIFO_DEPTH),
        .WIDTH                  (RSP_FIFO_WIDTH))
    i_rsp_fifo (
        .CLK                    (s00_axi_aclk),
        .RESET                  (~s00_axi_aresetn),

        // Read port
        .REN                    (rsp_fifo_ren),
        .RVALID                 (rsp_fifo_rvalid),
        .RDATA                  (rsp_fifo_rdata),
        .RCOUNT                 (rsp_fifo_rcount),
        .RERR                   (rsp_fifo_rerr),
        .ALMOST_EMPTY           (rsp_fifo_almost_empty),
        .EMPTY                  (rsp_fifo_empty),

        // Write port
        .WEN                    (rsp_fifo_wen),
        .WDATA                  (rsp_fifo_wdata),
        .WCOUNT                 (rsp_fifo_wcount),
        .WERR                   (rsp_fifo_werr),
        .ALMOST_FULL            (rsp_fifo_almost_full),
        .FULL                   (rsp_fifo_full)
    );


    // ------------------------------------------------------------------------
    // Core
    // ------------------------------------------------------------------------
    // Debug mux
    reg         [RSP_FIFO_WIDTH - 1 : 0] debug_y;
    reg                                  debug_y_dv;
    reg         [RSP_FIFO_WIDTH - 1 : 0] debug_y_pp;
    reg                                  debug_y_dv_pp;

    always @(*)
    begin
        case (rsp_fifo_sig_sel)
            4'd0:
            begin
                debug_y = cmd_fifo_rdata[RSP_FIFO_WIDTH - 1 : 0];
                debug_y_dv = cmd_fifo_rvalid;
            end

            default:
            begin
                debug_y = cmd_fifo_rdata[RSP_FIFO_WIDTH - 1 : 0];
                debug_y_dv = cmd_fifo_rvalid;
            end
        endcase
    end

    always @(posedge s00_axi_aclk)
    begin
        if (s00_axi_aresetn == 1'b0)
        begin
            debug_y_pp <= 'd0;
            debug_y_dv_pp <= 1'b0;
        end
        else
        begin
            debug_y_pp <= debug_y;
            debug_y_dv_pp <= debug_y_dv;
        end
    end

    assign cmd_fifo_ren = !cmd_fifo_empty && !cmd_fifo_hold;
    assign rsp_fifo_wen = !rsp_fifo_full && debug_y_dv_pp;
    assign rsp_fifo_wdata = debug_y_pp;


    // GPIO
    reg  [GPI_N_BITS - 1 : 0] gpi_sync0;
    reg  [GPI_N_BITS - 1 : 0] gpi_sync1;
    wire [GPI_N_BITS - 1 : 0] gpi_int;
    wire [GPO_N_BITS - 1 : 0] gpo_int;

    // GPI bits have no interdependence so its ok to sync independently
    always @(posedge s00_axi_aclk)
    begin
        if (s00_axi_aresetn == 1'b0)
        begin
            gpi_sync0 <= 'd0;
            gpi_sync1 <= 'd0;
        end
        else
        begin
            gpi_sync0 <= gpi_int;
            gpi_sync1 <= gpi_sync0;
        end
    end

    assign gpi_status = gpi_sync1;
    assign gpo_int = active_mem_u[GPO_WORD][GPO_N_BITS - 1 : 0];


    // ------------------------------------------------------------------------
    // Output buffers
    // ------------------------------------------------------------------------
    bcuda_ip1_io #(
        .BYPASS                 (0),
        .GPI_N_BITS             (GPI_N_BITS),
        .GPO_N_BITS             (GPO_N_BITS))
    i_io (
        .OEN                    (1'b1),

        // fabric side
        .GPO_INT                (gpo_int),

        .GPI_INT                (gpi_int),

        // pad side
        .GPI                    (GPI),

        .GPO                    (GPO)
    );


endmodule
