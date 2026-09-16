`include "delorean_defines.v"
`timescale 1 ns / 1 ps

module delorean_lcm #(
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
    input   wire    [C_S00_AXI_ADDR_WIDTH - 1 : 0]      s00_axi_awaddr,
    input   wire    [2 : 0]                             s00_axi_awprot,
    input   wire                                        s00_axi_awvalid,
    output  wire                                        s00_axi_awready,
    //      (W) Write Data channel
    input   wire    [C_S00_AXI_DATA_WIDTH - 1 : 0]      s00_axi_wdata,
    input   wire    [C_S00_AXI_DATA_WIDTH / 8 - 1 : 0]  s00_axi_wstrb,
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

    // DMA BRAM connections
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 s00_bram ADDR" *)
    input   wire    [11 : 0]                            s00_bram_addr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 s00_bram CLK" *)
    input   wire                                        s00_bram_clk,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 s00_bram DIN" *)
    input   wire    [C_S00_AXI_DATA_WIDTH - 1 : 0]      s00_bram_din,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 s00_bram EN" *)
    input   wire                                        s00_bram_en,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 s00_bram RST" *)
    input   wire                                        s00_bram_rst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 s00_bram WE" *)
    input   wire    [C_S00_AXI_DATA_WIDTH / 8 - 1 : 0]  s00_bram_we,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 s00_bram DOUT" *)
    output  wire    [C_S00_AXI_DATA_WIDTH - 1 : 0]      s00_bram_dout,

    // Clock for ODDR
    input   wire                                        CLK_90,

    // I/O connections
//    input   wire                                        RX_EIO2,
//    input   wire                                        RX_TP2,
    output  wire                                        RX_LVDS_CLK_P,
    output  wire                                        RX_LVDS_CLK_N,
    output  wire    [5 : 0]                             RX_LVDS_DATA_P,
    output  wire    [5 : 0]                             RX_LVDS_DATA_N,
    output  wire                                        RX_POL,
    output  wire                                        RX_TP1,

//    input   wire                                        TX_EIO2,
//    input   wire                                        TX_TP2,
    output  wire                                        TX_LVDS_CLK_P,
    output  wire                                        TX_LVDS_CLK_N,
    output  wire    [5 : 0]                             TX_LVDS_DATA_P,
    output  wire    [5 : 0]                             TX_LVDS_DATA_N,
    output  wire                                        TX_POL,
    output  wire                                        TX_TP1,

    output  wire                                        ITO_CLK,
    output  wire                                        LCD_EN,
    output  wire                                        PROG_TRIGGER,

    output  wire    [1 : 0]                             LASER_DR1_P,
    output  wire    [1 : 0]                             LASER_DR1_N,
    output  wire    [1 : 0]                             LASER_DR2_P,
    output  wire    [1 : 0]                             LASER_DR2_N,
    output  wire                                        LASER_TRIGGER,
    output  wire                                        TX_PWR_EN,
    output  wire                                        TX_PWR_SWITCH
);

    // Parameters: BRAM
    localparam integer BRAM_N_BUFFERS = 2;
    localparam integer BRAM_DEPTH_N_WORDS = 1024;
    localparam integer BRAM_WIDTH_N_BYTES = C_S00_AXI_DATA_WIDTH / 8;
    localparam integer BRAM_ADDR_N_BITS = $clog2(BRAM_DEPTH_N_WORDS);
    localparam integer BRAM_ADDR_LSB = $clog2(BRAM_WIDTH_N_BYTES);

    // Parameters: Register Memory
    localparam integer MEM_N_BITS = C_S00_AXI_DATA_DEPTH * C_S00_AXI_DATA_WIDTH;
    localparam integer MEM_N_STATUS_WORDS = C_S00_AXI_DATA_DEPTH / 4;
    localparam integer MEM_N_STATUS_BITS = MEM_N_STATUS_WORDS * C_S00_AXI_DATA_WIDTH;

    // map fields
    localparam integer TP1_PERIOD_N_BITS = 24;
    localparam integer RESET_CODE_N_BITS = 8;
    localparam integer AUX_CODE_ODD_N_BITS = 8;
    localparam integer AUX_CODE_EVEN_N_BITS = 8;
    localparam integer ITO_TC_N_BITS = 18;
    localparam integer STEP_N_BITS = 8;
    localparam integer RST_PW_N_BITS = 4;
    localparam integer TX_WAIT_N_BITS = 4;
    localparam integer TP1_PW_N_BITS = 8;
    localparam integer LASER_PW_SEL_N_BITS = 16;
    localparam integer CLKS_PER_INTERVAL_N_BITS = 8;
    localparam integer PULSES_PER_FRAME_N_BITS = 8;
    localparam integer INTERVALS_PER_FRAME_N_BITS = 16;
    localparam integer TCON_STATE_ONEHOT_N_BITS = C_S00_AXI_DATA_WIDTH;


    // ------------------------------------------------------------------------
    // AXI4-Lite controller
    // ------------------------------------------------------------------------
    wire                                            axi_wr_ack;
    wire                                            axi_wr_req;
    wire    [C_S00_AXI_ADDR_WIDTH - 1 : 0]          axi_wr_addr;
    wire    [C_S00_AXI_DATA_WIDTH - 1 : 0]          axi_wr_data;
    wire    [C_S00_AXI_DATA_WIDTH / 8 - 1 : 0]      axi_wr_strb;

    wire                                            axi_rd_valid;
    wire    [C_S00_AXI_DATA_WIDTH - 1  : 0]         axi_rd_data;
    wire                                            axi_rd_en;
    wire    [C_S00_AXI_ADDR_WIDTH - 1 : 0]          axi_rd_addr;


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
    // Registers
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


    // pack/unpack memory
    genvar i_gv;
    wire [C_S00_AXI_DATA_WIDTH - 1 : 0] reg_mem_u [C_S00_AXI_DATA_DEPTH - 1 : 0];
    wire [C_S00_AXI_DATA_WIDTH - 1 : 0] reg_mem_status_u [MEM_N_STATUS_WORDS - 1 : 0];

    generate
        for (i_gv = 0; i_gv < C_S00_AXI_DATA_DEPTH; i_gv = i_gv + 1)
        begin : gen_unpack
            assign reg_mem_u[i_gv] = reg_mem[i_gv * C_S00_AXI_DATA_WIDTH +:
                                             C_S00_AXI_DATA_WIDTH];
        end
    endgenerate

    generate
        for (i_gv = 0; i_gv < MEM_N_STATUS_WORDS; i_gv = i_gv + 1)
        begin : gen_pack
            assign reg_mem_status[i_gv * C_S00_AXI_DATA_WIDTH +: C_S00_AXI_DATA_WIDTH] =
                reg_mem_status_u[i_gv];
        end
    endgenerate


    // ------------------------------------------------------------------------
    // Memory map and miscellaneous control
    // ------------------------------------------------------------------------
    // Config and Control
    wire                                        mm_lcd_en;
    wire                                        mm_tcon_reset;
    wire                                        mm_tcon_enable;
    wire [BRAM_N_BUFFERS - 1 : 0]               mm_apply;
    wire [TP1_PERIOD_N_BITS - 1 : 0]            mm_tp1_period;
    wire [RESET_CODE_N_BITS - 1 : 0]            mm_reset_code;
    wire                                        mm_pol_finish_ovr;
    wire                                        mm_tp1_done_high;
    wire [AUX_CODE_EVEN_N_BITS - 1 : 0]         mm_aux_code_even;
    wire [AUX_CODE_ODD_N_BITS - 1 : 0]          mm_aux_code_odd;
    wire [ITO_TC_N_BITS - 1 : 0]                mm_ito_tc;
    wire                                        mm_ito_invert;
    wire                                        mm_ito_async;
    wire [STEP_N_BITS - 1 : 0]                  mm_n_steps;
    wire [RST_PW_N_BITS - 1 : 0]                mm_rst_pw;
    wire [TX_WAIT_N_BITS - 1 : 0]               mm_tx_wait;
    wire [TP1_PW_N_BITS - 1 : 0]                mm_tp1_pw;
    wire                                        mm_pol_ovr_en;
    wire                                        mm_pol_ovr_val;
    wire                                        mm_prog_trigger_mode;
    wire                                        mm_laser_enable;
    wire                                        mm_laser_start;
    wire [LASER_PW_SEL_N_BITS - 1 : 0]          mm_laser_pw_sel;
    wire [CLKS_PER_INTERVAL_N_BITS - 1 : 0]     mm_clks_per_interval;
    wire [PULSES_PER_FRAME_N_BITS - 1 : 0]      mm_pulses_per_frame;
    wire [INTERVALS_PER_FRAME_N_BITS - 1 : 0]   mm_intervals_per_frame;
    wire                                        mm_tx_pwr_en;
    wire                                        mm_tx_pwr_switch;

    assign mm_lcd_en                = reg_mem_u[0][0];
    assign mm_tcon_reset            = reg_mem_u[1][0];
    assign mm_tcon_enable           = reg_mem_u[1][1];
    assign mm_apply[0]              = reg_mem_wr_valids[2];
    assign mm_apply[1]              = reg_mem_wr_valids[3];
    assign mm_tp1_period            = reg_mem_u[4][TP1_PERIOD_N_BITS - 1 : 0];
    assign mm_reset_code            = reg_mem_u[5][RESET_CODE_N_BITS - 1 : 0];
    assign mm_pol_finish_ovr        = reg_mem_u[5][RESET_CODE_N_BITS];
    assign mm_tp1_done_high         = reg_mem_u[5][RESET_CODE_N_BITS + 1];
    assign mm_aux_code_even         = reg_mem_u[6][0 +: AUX_CODE_EVEN_N_BITS];
    assign mm_aux_code_odd          = reg_mem_u[6][AUX_CODE_EVEN_N_BITS +: AUX_CODE_ODD_N_BITS];
    assign mm_ito_tc                = reg_mem_u[7][ITO_TC_N_BITS - 1 : 0];
    assign mm_ito_invert            = reg_mem_u[7][ITO_TC_N_BITS];
    assign mm_ito_async             = reg_mem_u[7][ITO_TC_N_BITS + 1];
    assign mm_n_steps               = reg_mem_u[8][0 +: STEP_N_BITS];
    assign mm_rst_pw                = reg_mem_u[8][STEP_N_BITS +: RST_PW_N_BITS];
    assign mm_tx_wait               = reg_mem_u[8][STEP_N_BITS + RST_PW_N_BITS +: TX_WAIT_N_BITS];
    assign mm_tp1_pw                = reg_mem_u[8][STEP_N_BITS + RST_PW_N_BITS + TX_WAIT_N_BITS +: TP1_PW_N_BITS];
    assign mm_pol_ovr_en            = reg_mem_u[9][0];
    assign mm_pol_ovr_val           = reg_mem_u[9][1];
    assign mm_prog_trigger_mode     = reg_mem_u[10][0];
    assign mm_laser_enable          = reg_mem_u[12][0];
    assign mm_laser_start           = reg_mem_u[12][0] & reg_mem_wr_valids[12];
    assign mm_laser_pw_sel          = reg_mem_u[13][LASER_PW_SEL_N_BITS - 1 : 0];
    assign mm_clks_per_interval     = reg_mem_u[15][CLKS_PER_INTERVAL_N_BITS - 1 : 0];
    assign mm_pulses_per_frame      = reg_mem_u[16][PULSES_PER_FRAME_N_BITS - 1 : 0];
    assign mm_intervals_per_frame   = reg_mem_u[17][INTERVALS_PER_FRAME_N_BITS - 1 : 0];
    assign mm_tx_pwr_en             = reg_mem_u[18][0];
    assign mm_tx_pwr_switch         = reg_mem_u[18][1];

    // Status
    wire [C_S00_AXI_DATA_WIDTH - 1 : 0]     clk_freq;
    wire [BRAM_N_BUFFERS - 1 : 0]           loading;
    wire [TCON_STATE_ONEHOT_N_BITS - 1 : 0] tcon_state_onehot;
    wire [BRAM_N_BUFFERS - 1 : 0]           apply_cache;

    assign clk_freq = 'd100;

    assign reg_mem_status_u[0] = clk_freq;
    assign reg_mem_status_u[1] = loading;
    assign reg_mem_status_u[2] = tcon_state_onehot;
    assign reg_mem_status_u[3] = apply_cache;
    generate
        for (i_gv = 4; i_gv < MEM_N_STATUS_WORDS; i_gv = i_gv + 1)
        begin : gen_zerofill
            assign reg_mem_status_u[i_gv] = 'd0;
        end
    endgenerate


    // ------------------------------------------------------------------------
    // BRAM for DMA
    // ------------------------------------------------------------------------
    wire [BRAM_ADDR_N_BITS - 1 : 0]         bram_addr_byte2word;
    wire [BRAM_ADDR_N_BITS - 1 : 0]         dma_bram_addr;
    wire                                    dma_bram_clk;
    wire [C_S00_AXI_DATA_WIDTH - 1 : 0]     dma_bram_din;
    wire                                    dma_bram_en;
    wire [C_S00_AXI_DATA_WIDTH / 8 - 1 : 0] dma_bram_we;
    wire [C_S00_AXI_DATA_WIDTH - 1 : 0]     dma_bram_dout;

    assign bram_addr_byte2word = s00_bram_addr[BRAM_ADDR_LSB +: BRAM_ADDR_N_BITS];

    delorean_dma_bram U_delorean_dma_bram (
        .A_ADDR                 (bram_addr_byte2word),
        .A_CLK                  (s00_bram_clk),
        .A_DI                   (s00_bram_din),
        .A_EN                   (s00_bram_en),
        .A_WE                   (s00_bram_we),
        .A_DO                   (s00_bram_dout),

        .B_ADDR                 (dma_bram_addr),
        .B_CLK                  (dma_bram_clk),
        .B_DI                   (dma_bram_din),
        .B_EN                   (dma_bram_en),
        .B_WE                   (dma_bram_we),
        .B_DO                   (dma_bram_dout)
    );


    // ------------------------------------------------------------------------
    // LCM timing controller
    // ------------------------------------------------------------------------
    // TCON - MLVDS connections
    wire [$clog2(BRAM_N_BUFFERS) - 1 : 0]   buf_idx;
    wire                                    prog_start;
    wire                                    use_reset_code;
    wire                                    prog_done;
    // IP Outputs
    wire                                    ito_clk_int;
    wire                                    pol_int;
    wire                                    prog_trigger_int;
    wire                                    tp1_int;

    delorean_tcon #(
        .ITO_CNT_N_BITS         (ITO_TC_N_BITS),
        .N_BUFFERS              (BRAM_N_BUFFERS),
        .TP1_PERIOD_N_BITS      (TP1_PERIOD_N_BITS),
        .TP1_PW_N_BITS          (TP1_PW_N_BITS))
    U_delorean_tcon (
        .APPLY                  (mm_apply),
        .CLK                    (s00_axi_aclk),
        .ENABLE                 (mm_tcon_enable),
        .ITO_ASYNC              (mm_ito_async),
        .ITO_INVERT             (mm_ito_invert),
        .ITO_TC                 (mm_ito_tc),
        .POL_FINISH_OVR         (mm_pol_finish_ovr),
        .POL_OVR_EN             (mm_pol_ovr_en),
        .POL_OVR_VAL            (mm_pol_ovr_val),
        .PROG_TRIGGER_MODE      (mm_prog_trigger_mode),
        .PROG_DONE              (prog_done),
        .RESET                  (mm_tcon_reset || ~s00_axi_aresetn),
        .TP1_DONE_HIGH          (mm_tp1_done_high),
        .TP1_PERIOD             (mm_tp1_period),
        .TP1_PW                 (mm_tp1_pw),

        .APPLY_CACHE            (apply_cache),
        .BUF_IDX                (buf_idx),
        .ITO_CLK                (ito_clk_int),
        .LOADING                (loading),
        .POL                    (pol_int),
        .PROG_START             (prog_start),
        .PROG_TRIGGER           (prog_trigger_int),
        .STATE_ONEHOT           (tcon_state_onehot),
        .TP1                    (tp1_int),
        .USE_RESET_CODE         (use_reset_code)
    );


    // ------------------------------------------------------------------------
    // Mini-LVDS transmitter
    // ------------------------------------------------------------------------
    wire [5 : 0]    rx_lvds_data_d1_int;
    wire [5 : 0]    rx_lvds_data_d2_int;
    wire [5 : 0]    tx_lvds_data_d1_int;
    wire [5 : 0]    tx_lvds_data_d2_int;

    delorean_mlvds #(
        .C_S_AXI_DATA_WIDTH     (C_S00_AXI_DATA_WIDTH),
        .N_BUFFERS              (BRAM_N_BUFFERS),
        .RST_PW_N_BITS          (RST_PW_N_BITS),
        .STEP_N_BITS            (STEP_N_BITS),
        .TX_WAIT_N_BITS         (TX_WAIT_N_BITS))
    U_delorean_mlvds (
        // BRAM interface
        .BRAM_DO                (dma_bram_dout),
        .BRAM_ADDR              (dma_bram_addr),
        .BRAM_CLK               (dma_bram_clk),
        .BRAM_DI                (dma_bram_din),
        .BRAM_EN                (dma_bram_en),
        .BRAM_WE                (dma_bram_we),

        // non-BRAM interface
        .AUX_CODE_EVEN          (mm_aux_code_even),
        .AUX_CODE_ODD           (mm_aux_code_odd),
        .BUF_IDX                (buf_idx),
        .CLK                    (s00_axi_aclk),
        .N_STEPS                (mm_n_steps),
        .RESET                  (mm_tcon_reset || ~s00_axi_aresetn),
        .RESET_CODE             (mm_reset_code),
        .RST_PW                 (mm_rst_pw),
        .START                  (prog_start),
        .TX_WAIT                (mm_tx_wait),
        .USE_RESET_CODE         (use_reset_code),

        .DONE                   (prog_done),
        .RX_LVDS_DATA_D1        (rx_lvds_data_d1_int),
        .RX_LVDS_DATA_D2        (rx_lvds_data_d2_int),
        .TX_LVDS_DATA_D1        (tx_lvds_data_d1_int),
        .TX_LVDS_DATA_D2        (tx_lvds_data_d2_int)
    );


    // ------------------------------------------------------------------------
    // Laser Control
    // ------------------------------------------------------------------------
    wire laser_dr1_int;
    wire laser_dr2_int;
    wire laser_trigger_int;

    delorean_laser_control #(
        .LASER_PW_SEL_N_BITS        (LASER_PW_SEL_N_BITS),
        .CLKS_PER_INTERVAL_N_BITS   (CLKS_PER_INTERVAL_N_BITS),
        .PULSES_PER_FRAME_N_BITS    (PULSES_PER_FRAME_N_BITS),
        .INTERVALS_PER_FRAME_N_BITS (INTERVALS_PER_FRAME_N_BITS))
    U_delorean_laser_control(
        .S_AXI_ACLK                 (s00_axi_aclk),
        .S_AXI_ARESETN              (s00_axi_aresetn),
        .LASER_ENABLE               (mm_laser_enable),
        .LASER_START                (mm_laser_start),
        .LASER_PW_SEL               (mm_laser_pw_sel),
        .CLKS_PER_INTERVAL          (mm_clks_per_interval),
        .PULSES_PER_FRAME           (mm_pulses_per_frame),
        .INTERVALS_PER_FRAME        (mm_intervals_per_frame),

        .LASER_DR1                  (laser_dr1_int),
        .LASER_DR2                  (laser_dr2_int),
        .LASER_TRIGGER              (laser_trigger_int)
    );


    // ------------------------------------------------------------------------
    // Output buffers
    // ------------------------------------------------------------------------
    delorean_lcm_io_mlvds U_delorean_lcm_io_mlvds_rx (
        .CLK                    (s00_axi_aclk),
        .CLK_90                 (CLK_90),

        // Mini-LVDS
        .LVDS_DATA_D1           (rx_lvds_data_d1_int),
        .LVDS_DATA_D2           (rx_lvds_data_d2_int),
        .POL_INT                (pol_int),
        .TP1_INT                (tp1_int),

        .LVDS_CLK_P             (RX_LVDS_CLK_P),
        .LVDS_CLK_N             (RX_LVDS_CLK_N),
        .LVDS_DATA_P            (RX_LVDS_DATA_P),
        .LVDS_DATA_N            (RX_LVDS_DATA_N),
        .POL                    (RX_POL),
        .TP1                    (RX_TP1),

        // GPO
        .ITO_CLK_INT            (ito_clk_int),
        .LCD_EN_INT             (mm_lcd_en),
        .PROG_TRIGGER_INT       (prog_trigger_int),

        .ITO_CLK                (ITO_CLK),
        .LCD_EN                 (LCD_EN),
        .PROG_TRIGGER           (PROG_TRIGGER)
    );


    delorean_lcm_io_mlvds U_delorean_lcm_io_mlvds_tx (
        .CLK                    (s00_axi_aclk),
        .CLK_90                 (CLK_90),

        // Mini-LVDS
        .LVDS_DATA_D1           (tx_lvds_data_d1_int),
        .LVDS_DATA_D2           (tx_lvds_data_d2_int),
        .POL_INT                (pol_int),
        .TP1_INT                (tp1_int),

        .LVDS_CLK_P             (TX_LVDS_CLK_P),
        .LVDS_CLK_N             (TX_LVDS_CLK_N),
        .LVDS_DATA_P            (TX_LVDS_DATA_P),
        .LVDS_DATA_N            (TX_LVDS_DATA_N),
        .POL                    (TX_POL),
        .TP1                    (TX_TP1),

        // GPO
        .ITO_CLK_INT            (ito_clk_int),
        .LCD_EN_INT             (mm_lcd_en),
        .PROG_TRIGGER_INT       (prog_trigger_int),

        .ITO_CLK                (),
        .LCD_EN                 (),
        .PROG_TRIGGER           ()
    );


    delorean_lcm_io_laser U_delorean_lcm_io_laser (
        .OUT_EN                 (1'b1),
        .LASER_DR1_INT          (laser_dr1_int),
        .LASER_DR2_INT          (laser_dr2_int),
        .LASER_TRIGGER_INT      (laser_trigger_int),
        .TX_PWR_EN_INT          (mm_tx_pwr_en),
        .TX_PWR_SWITCH_INT      (mm_tx_pwr_switch),

        .LASER_DR1_P            (LASER_DR1_P),
        .LASER_DR1_N            (LASER_DR1_N),
        .LASER_DR2_P            (LASER_DR2_P),
        .LASER_DR2_N            (LASER_DR2_N),
        .LASER_TRIGGER          (LASER_TRIGGER),
        .TX_PWR_EN              (TX_PWR_EN),
        .TX_PWR_SWITCH          (TX_PWR_SWITCH)
    );

endmodule
