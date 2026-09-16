`include "delorean_defines.v"
`timescale 1 ns / 1 ps

module delorean_spi #(
    parameter integer C_S00_AXI_DATA_DEPTH = 32,
    parameter integer C_S00_AXI_DATA_WIDTH = 32,
    parameter integer C_S00_AXI_ADDR_WIDTH = 12
)
(
    // AXI connections
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
    input   wire    [1 : 0]                             SPI_MISO,
    output  wire    [1 : 0]                             SPI_ADC_CH_SEL,
    output  wire                                        SPI_DAISY_EN,
    output  wire    [1 : 0]                             SPI_MOSI,
    output  wire    [1 : 0]                             SPI_SCLK,
    output  wire    [5 : 0]                             SPI_CS_B
);


    // Parameters: SPI
    localparam integer SPI_N_SLAVES = 6;
    localparam integer SPI_MODE = 2; // 2SPI
    localparam integer SPI_BASE_FRAME_N_BITS = 16; // must be a power of 2
    localparam integer SPI_MAX_BASE_FRAMES = 16;
    localparam integer SPI_FRAME_MAX_N_BITS = SPI_BASE_FRAME_N_BITS * SPI_MAX_BASE_FRAMES;

    // Parameters: Register Memory
    localparam integer MEM_N_BITS = C_S00_AXI_DATA_DEPTH * C_S00_AXI_DATA_WIDTH;
    localparam integer MEM_N_STATUS_WORDS = 5;
    localparam integer MEM_N_STATUS_BITS = MEM_N_STATUS_WORDS * C_S00_AXI_DATA_WIDTH;

    // Parameters: FIFOs
    localparam integer CMD_FIFO_WIDTH = C_S00_AXI_DATA_WIDTH;
    localparam integer CMD_FIFO_DEPTH = 16;
    localparam integer CMD_FIFO_COUNT_N_BITS = $clog2(CMD_FIFO_DEPTH);

    localparam integer RSP_FIFO_WIDTH = SPI_BASE_FRAME_N_BITS + $clog2(SPI_N_SLAVES);
    localparam integer RSP_FIFO_DEPTH = 64;
    localparam integer RSP_FIFO_COUNT_N_BITS = $clog2(RSP_FIFO_DEPTH);


    // Downstream write interfaces to memory
    wire                                            axi_wr_ack;
    wire                                            axi_wr_req;
    wire    [C_S00_AXI_ADDR_WIDTH - 1 : 0]          axi_wr_addr;
    wire    [C_S00_AXI_DATA_WIDTH - 1 : 0]          axi_wr_data;
    wire    [(C_S00_AXI_DATA_WIDTH / 8) - 1 : 0]    axi_wr_strb;

    // Downstream read interfaces to memory
    wire                                            axi_rd_valid;
    wire    [C_S00_AXI_DATA_WIDTH - 1  : 0]         axi_rd_data;
    wire                                            axi_rd_en;
    wire    [C_S00_AXI_ADDR_WIDTH - 1 : 0]          axi_rd_addr;

    // FIFO interfaces
    wire                                            cmd_fifo_ren;
    wire    [CMD_FIFO_WIDTH - 1 : 0]                cmd_fifo_rdata;
    wire                                            cmd_fifo_empty;
    wire    [CMD_FIFO_COUNT_N_BITS - 1 : 0]         cmd_fifo_count;
    wire                                            cmd_fifo_wen;
    wire    [CMD_FIFO_WIDTH - 1 : 0]                cmd_fifo_wdata;
    wire                                            cmd_fifo_full;

    wire                                            rsp_fifo_reset[1:0];
    wire                                            rsp_fifo_ren[1:0];
    wire    [RSP_FIFO_WIDTH - 1 : 0]                rsp_fifo_rdata[1:0];
    wire                                            rsp_fifo_empty[1:0];
    wire    [RSP_FIFO_COUNT_N_BITS - 1 : 0]         rsp_fifo_count[1:0];
    wire                                            rsp_fifo_wen[1:0];
    wire    [RSP_FIFO_WIDTH - 1 : 0]                rsp_fifo_wdata[1:0];
    wire                                            rsp_fifo_full[1:0];


    // ------------------------------------------------------------------------
    // AXI4-Lite controller
    // ------------------------------------------------------------------------
    delorean_axi_interface #(
        .C_S_AXI_DATA_WIDTH     (C_S00_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH     (C_S00_AXI_ADDR_WIDTH))
    U_delorean_axi_interface (
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
    // Register file
    // ------------------------------------------------------------------------
    wire [MEM_N_BITS - 1 : 0]           reg_mem;
    wire [C_S00_AXI_DATA_DEPTH - 1 : 0] reg_mem_rd_valids;
    wire [C_S00_AXI_DATA_DEPTH - 1 : 0] reg_mem_wr_valids;
    wire [MEM_N_STATUS_BITS - 1 : 0]    reg_mem_status;
    wire                                reg_mem_wr_back_pressure;

    assign reg_mem_wr_back_pressure = 1'b0;

    delorean_regs #(
        .C_S_AXI_DATA_DEPTH     (C_S00_AXI_DATA_DEPTH),
        .C_S_AXI_DATA_WIDTH     (C_S00_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH     (C_S00_AXI_ADDR_WIDTH),
        .N_STATUS_WORDS         (MEM_N_STATUS_WORDS))
    U_delorean_regs (
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
        .STATUS                 (reg_mem_status),
        .WR_BACK_PRESSURE       (reg_mem_wr_back_pressure),
        .MEMORY                 (reg_mem),
        .READ_VALIDS            (reg_mem_rd_valids),
        .WRITE_VALIDS           (reg_mem_wr_valids)
    );


    // ------------------------------------------------------------------------
    // Memory map and miscellaneous control
    // ------------------------------------------------------------------------
    localparam integer SPI_RSP_WORD_1 = C_S00_AXI_DATA_DEPTH - 3;
    localparam integer SPI_RSP_WORD_0 = C_S00_AXI_DATA_DEPTH - 2;
    localparam integer SPI_CMD_WORD = C_S00_AXI_DATA_DEPTH - 6;
    localparam integer SPI_CONTROL_WORD = C_S00_AXI_DATA_DEPTH - 7;
    localparam integer SPI_CLK_WORD = C_S00_AXI_DATA_DEPTH - 8;

    localparam integer SPI_TABLE_OFFSET = 0;
    localparam integer SPI_TABLE_N_BITS = SPI_MODE * SPI_FRAME_MAX_N_BITS;

    // unpack the memory
    wire [C_S00_AXI_DATA_WIDTH - 1 : 0] reg_mem_u [C_S00_AXI_DATA_DEPTH - 1 : 0];

    genvar i_gv;
    generate
        for( i_gv = 0; i_gv < C_S00_AXI_DATA_DEPTH; i_gv = i_gv + 1 )
        begin : unpack
            assign reg_mem_u[i_gv] = reg_mem[C_S00_AXI_DATA_WIDTH * (i_gv+1) - 1 :
                                             C_S00_AXI_DATA_WIDTH * i_gv];
        end
    endgenerate


    // Status
    wire [C_S00_AXI_DATA_WIDTH - 1 : 0] status_word;
    wire [C_S00_AXI_DATA_WIDTH - 1 : 0] rsp_word[1:0];

    assign reg_mem_status = {status_word,
                             rsp_word[0],
                             rsp_word[1],
                             32'd0,
                             32'd0};

    assign status_word = {cmd_fifo_full, cmd_fifo_count};

    assign rsp_word[0] = {!rsp_fifo_empty[0],
                          6'd0,
                          rsp_fifo_count[0],
                          rsp_fifo_rdata[0]};

    assign rsp_word[1] = {!rsp_fifo_empty[1],
                          6'd0,
                          rsp_fifo_count[1],
                          rsp_fifo_rdata[1]};

    // Config/Control
    // write fifo after data has been written to regfile
    assign cmd_fifo_wen = reg_mem_wr_valids[SPI_CMD_WORD];
    assign cmd_fifo_wdata = reg_mem_u[SPI_CMD_WORD];

    // rd_valid indicates the FIFO output is being read in this cycle
    assign rsp_fifo_ren[0] = reg_mem_rd_valids[SPI_RSP_WORD_0];
    assign rsp_fifo_ren[1] = reg_mem_rd_valids[SPI_RSP_WORD_1];

    // allow clearing RSP FIFOs on writing
    assign rsp_fifo_reset[0] = reg_mem_wr_valids[SPI_RSP_WORD_0];
    assign rsp_fifo_reset[1] = reg_mem_wr_valids[SPI_RSP_WORD_1];

    // SPI
    wire    [C_S00_AXI_DATA_WIDTH - 1 : 0]      spi_control;
    wire    [C_S00_AXI_DATA_WIDTH - 1 : 0]      spi_clk_config;
    wire    [SPI_TABLE_N_BITS - 1 : 0]          spi_table;

    assign spi_control = reg_mem_u[SPI_CONTROL_WORD];
    assign spi_clk_config = reg_mem_u[SPI_CLK_WORD];
    assign spi_table = reg_mem[SPI_TABLE_OFFSET +: SPI_TABLE_N_BITS];


    // ------------------------------------------------------------------------
    // FIFOs
    // ------------------------------------------------------------------------
    delorean_fifo #(
        .DEPTH                  (CMD_FIFO_DEPTH),
        .WIDTH                  (CMD_FIFO_WIDTH))
    U_delorean_cmd_fifo (
        .S_AXI_ACLK             (s00_axi_aclk),
        .RESET                  (~s00_axi_aresetn),

        // Read port
        .REN                    (cmd_fifo_ren),
        .RDATA                  (cmd_fifo_rdata),
        .EMPTY                  (cmd_fifo_empty),
        .COUNT                  (cmd_fifo_count),

        // Write port
        .WEN                    (cmd_fifo_wen),
        .WDATA                  (cmd_fifo_wdata),
        .FULL                   (cmd_fifo_full)
    );

    generate
        for( i_gv = 0; i_gv < 2; i_gv = i_gv + 1 )
        begin : gen_fifo
            delorean_fifo #(
                .DEPTH                  (RSP_FIFO_DEPTH),
                .WIDTH                  (RSP_FIFO_WIDTH))
            U_delorean_rsp_fifo (
                .S_AXI_ACLK             (s00_axi_aclk),
                .RESET                  (~s00_axi_aresetn || rsp_fifo_reset[i_gv]),

                // Read port
                .REN                    (rsp_fifo_ren[i_gv]),
                .RDATA                  (rsp_fifo_rdata[i_gv]),
                .EMPTY                  (rsp_fifo_empty[i_gv]),
                .COUNT                  (rsp_fifo_count[i_gv]),

                // Write port
                .WEN                    (rsp_fifo_wen[i_gv]),
                .WDATA                  (rsp_fifo_wdata[i_gv]),
                .FULL                   (rsp_fifo_full[i_gv])
            );
        end
    endgenerate


    // ------------------------------------------------------------------------
    // SPI FSM
    // ------------------------------------------------------------------------
    localparam integer CLK_DIV_N_BITS = 5;

    wire    [1 : 0]                                 spi_adc_chsel_int;
    wire    [CLK_DIV_N_BITS - 1 : 0]                spi_clk_div;
    wire                                            spi_daisy_en_int;
    wire    [$clog2(SPI_MAX_BASE_FRAMES) - 1 : 0]   spi_n_base_frames;
    wire    [$clog2(SPI_N_SLAVES) - 1 : 0]          spi_slave_idx;
    wire                                            spi_start;
    wire    [SPI_BASE_FRAME_N_BITS - 1 : 0]         spi_wdata_0;
    wire    [SPI_BASE_FRAME_N_BITS - 1 : 0]         spi_wdata_1;

    wire                                            spi_done;
    wire    [SPI_BASE_FRAME_N_BITS - 1 : 0]         spi_rdata_0;
    wire    [SPI_BASE_FRAME_N_BITS - 1 : 0]         spi_rdata_1;
    wire                                            spi_wdata_req;

    delorean_spi_fsm #(
        .CONTROL_N_BITS         (C_S00_AXI_DATA_WIDTH),
        .CLK_CONFIG_N_BITS      (C_S00_AXI_DATA_WIDTH),
        .TABLE_N_BITS           (SPI_TABLE_N_BITS),
        .CMD_FIFO_WIDTH         (CMD_FIFO_WIDTH),
        .RSP_FIFO_WIDTH         (RSP_FIFO_WIDTH),
        .RSP_FIFO_DEPTH         (RSP_FIFO_DEPTH),
        .BASE_FRAME_N_BITS      (SPI_BASE_FRAME_N_BITS),
        .CLK_DIV_N_BITS         (CLK_DIV_N_BITS),
        .MAX_BASE_FRAMES        (SPI_MAX_BASE_FRAMES),
        .N_SLAVES               (SPI_N_SLAVES))
    U_delorean_spi_fsm(
        .S_AXI_ACLK             (s00_axi_aclk),
        .S_AXI_ARESETN          (s00_axi_aresetn),

        // Config / Control
        .SPI_CONTROL            (spi_control),
        .SPI_CLK_CONFIG         (spi_clk_config),
        .SPI_TABLE              (spi_table),

        // FIFO side
        .CMD_FIFO_EMPTY         (cmd_fifo_empty),
        .CMD_FIFO_RDATA         (cmd_fifo_rdata),
        .CMD_FIFO_REN           (cmd_fifo_ren),

        .RSP_FIFO_FULL_0        (rsp_fifo_full[0]),
        .RSP_FIFO_FULL_1        (rsp_fifo_full[1]),
        .RSP_FIFO_WDATA_0       (rsp_fifo_wdata[0]),
        .RSP_FIFO_WDATA_1       (rsp_fifo_wdata[1]),
        .RSP_FIFO_WEN_0         (rsp_fifo_wen[0]),
        .RSP_FIFO_WEN_1         (rsp_fifo_wen[1]),

        // Controller side
        .DONE                   (spi_done),
        .RDATA_0                (spi_rdata_0),
        .RDATA_1                (spi_rdata_1),
        .WDATA_REQ              (spi_wdata_req),

        .ADC_CHSEL              (spi_adc_chsel_int),
        .CLK_DIV                (spi_clk_div),
        .DAISY_EN               (spi_daisy_en_int),
        .N_BASE_FRAMES          (spi_n_base_frames),
        .SLAVE_IDX              (spi_slave_idx),
        .START                  (spi_start),
        .WDATA_0                (spi_wdata_0),
        .WDATA_1                (spi_wdata_1)
    );


    // ------------------------------------------------------------------------
    // SPI controller
    // ------------------------------------------------------------------------
    wire    [SPI_MODE - 1 : 0]      spi_miso_int;
    wire    [SPI_MODE - 1 : 0]      spi_mosi_int;
    wire                            spi_sclk_int;
    wire    [SPI_N_SLAVES - 1 : 0]  spi_cs_b_int;

    delorean_spi_controller #(
        .BASE_FRAME_N_BITS      (SPI_BASE_FRAME_N_BITS),
        .CLK_DIV_N_BITS         (CLK_DIV_N_BITS),
        .MAX_BASE_FRAMES        (SPI_MAX_BASE_FRAMES),
        .N_SLAVES               (SPI_N_SLAVES))
    U_delorean_spi_controller(
        .S_AXI_ACLK             (s00_axi_aclk),
        .S_AXI_ARESETN          (s00_axi_aresetn),

        // Host side
        .CLK_DIV                (spi_clk_div),
        .N_BASE_FRAMES          (spi_n_base_frames),
        .SLAVE_IDX              (spi_slave_idx),
        .START                  (spi_start),
        .WDATA_0                (spi_wdata_0),
        .WDATA_1                (spi_wdata_1),

        .DONE                   (spi_done),
        .RDATA_0                (spi_rdata_0),
        .RDATA_1                (spi_rdata_1),
        .WDATA_REQ              (spi_wdata_req),

        // Peripheral side
        .MISO                   (spi_miso_int),
        .MOSI                   (spi_mosi_int),
        .SCLK                   (spi_sclk_int),
        .SS_B                   (spi_cs_b_int)
    );


    // ------------------------------------------------------------------------
    // I/O buffers
    // ------------------------------------------------------------------------
    delorean_spi_io #(
        .N_SLAVES               (SPI_N_SLAVES))
    U_delorean_spi_io(
        .OEN                    (1'b1),
        .ADC_CHSEL_INT          (spi_adc_chsel_int),
        .DAISY_EN_INT           (spi_daisy_en_int),
        .MOSI_INT               (spi_mosi_int),
        .SCLK_INT               (spi_sclk_int),
        .SS_B_INT               (spi_cs_b_int),
        .MISO_INT               (spi_miso_int),

        .MISO                   (SPI_MISO),
        .ADC_CHSEL              (SPI_ADC_CH_SEL),
        .DAISY_EN               (SPI_DAISY_EN),
        .MOSI                   (SPI_MOSI),
        .SCLK                   (SPI_SCLK),
        .SS_B                   (SPI_CS_B)
    );

endmodule
