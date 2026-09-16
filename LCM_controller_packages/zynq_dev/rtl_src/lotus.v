`include "lotus_defines.v"
`timescale 1 ns / 1 ps

module lotus #(
    parameter integer C_S00_AXI_DATA_DEPTH = 64,
    parameter integer C_S00_AXI_DATA_WIDTH = 32,
    parameter integer C_S00_AXI_ADDR_WIDTH = 9
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
    output  wire                                        TP1,

    output  wire                                        ITO_CLK,
    output  wire                                        STATE_EQ_PROG,

    output  wire                                        LASER_DR1_N,
    output  wire                                        LASER_DR1_P,
    output  wire                                        LASER_DR2_N,
    output  wire                                        LASER_DR2_P,
    output  wire                                        LASER_TRIGGER
);


    localparam integer N_BSC_COEFFS = 204;
    localparam integer COEFF_N_BITS = 8;
    localparam integer TABLE_N_BITS = N_BSC_COEFFS * COEFF_N_BITS;
    localparam integer TABLE_N_WORDS = TABLE_N_BITS / C_S00_AXI_DATA_WIDTH; // 51

    // Paramers: ROM
    localparam integer ROM_N_TABLES = 128;
    localparam integer ROM_DEPTH = ROM_N_TABLES * TABLE_N_WORDS; // 6528
    localparam integer ROM_WIDTH = C_S00_AXI_DATA_WIDTH;

    // Paramers: Active Memory
    localparam integer MEM_N_BITS = C_S00_AXI_DATA_DEPTH * C_S00_AXI_DATA_WIDTH;
    localparam integer MEM_N_STATUS_WORDS = 1;
    localparam integer MEM_N_STATUS_BITS = MEM_N_STATUS_WORDS * C_S00_AXI_DATA_WIDTH;


    // Downstream write interfaces to active memory
    wire                                            axi_wr_ack;
    wire                                            axi_wr_req;
    wire    [C_S00_AXI_ADDR_WIDTH - 1 : 0]          axi_wr_addr;
    wire    [C_S00_AXI_DATA_WIDTH - 1 : 0]          axi_wr_data;
    wire    [(C_S00_AXI_DATA_WIDTH / 8) - 1 : 0]    axi_wr_strb;

    wire                                            mvr_wr_ack;
    wire                                            mvr_wr_req;
    wire    [C_S00_AXI_ADDR_WIDTH - 1 : 0]          mvr_wr_addr;
    wire    [C_S00_AXI_DATA_WIDTH - 1 : 0]          mvr_wr_data;
    wire    [(C_S00_AXI_DATA_WIDTH / 8) - 1 : 0]    mvr_wr_strb;

    // Downstream read interfaces to active memory
    wire                                            axi_rd_valid;
    wire    [C_S00_AXI_DATA_WIDTH - 1  : 0]         axi_rd_data;
    wire                                            axi_rd_en;
    wire    [C_S00_AXI_ADDR_WIDTH - 1 : 0]          axi_rd_addr;

    // Downstream read interface to ROM
    //wire                                            mvr_rd_valid;
    wire    [C_S00_AXI_DATA_WIDTH - 1  : 0]         mvr_rd_data;
    wire                                            mvr_rd_en;
    wire    [$clog2(ROM_DEPTH) - 1 : 0]             mvr_rd_addr;

    // Table Mover control
    reg                                             mvr_start_txfer;
    wire    [$clog2(ROM_N_TABLES) - 1 : 0]          mvr_table_idx;
    wire                                            mvr_txfer_busy;
    wire                                            mvr_txfer_done;


    // ------------------------------------------------------------------------
    // AXI4-Lite controller
    // ------------------------------------------------------------------------
    lotus_axi_interface #(
        .C_S_AXI_DATA_WIDTH     (C_S00_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH     (C_S00_AXI_ADDR_WIDTH))
    I_lotus_axi_interface (
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
    // Table mover
    // ------------------------------------------------------------------------
    lotus_table_mover #(
        .C_S_AXI_DATA_WIDTH     (C_S00_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH     (C_S00_AXI_ADDR_WIDTH),
        .ROM_DEPTH              (ROM_DEPTH),
        .ROM_WIDTH              (ROM_WIDTH),
        .ROM_N_TABLES           (ROM_N_TABLES))
    I_lotus_table_mover (
        .S_AXI_ACLK             (s00_axi_aclk),
        .S_AXI_ARESETN          (s00_axi_aresetn),

        // Control
        .START_TXFER            (mvr_start_txfer),
        .TABLE_IDX              (mvr_table_idx),
        .TXFER_BUSY             (mvr_txfer_busy),
        .TXFER_DONE             (mvr_txfer_done),

        // RAM write interface (sink)
        .WR_ACK                 (mvr_wr_ack),
        .WR_REQ                 (mvr_wr_req),
        .WR_ADDR                (mvr_wr_addr),
        .WR_DATA                (mvr_wr_data),
        .WR_STRB                (mvr_wr_strb),

        // ROM read interface (source)
        .RD_DATA                (mvr_rd_data),
        .RD_EN                  (mvr_rd_en),
        .RD_ADDR                (mvr_rd_addr)
    );


    // ------------------------------------------------------------------------
    // ROM for steering patterns
    // ------------------------------------------------------------------------
    lotus_rom #(
        .DEPTH                  (ROM_DEPTH),
        .WIDTH                  (ROM_WIDTH))
    I_lotus_rom (
        .CLK                    (s00_axi_aclk),
        .EN                     (mvr_rd_en),
        .ADDR                   (mvr_rd_addr),
        .DATA                   (mvr_rd_data)
    );


    // ------------------------------------------------------------------------
    // Active Memory
    // ------------------------------------------------------------------------
    wire [MEM_N_BITS - 1 : 0]           active_mem;
    wire [C_S00_AXI_DATA_DEPTH - 1 : 0] active_mem_wr_enables;
    wire [MEM_N_STATUS_BITS - 1 : 0]    active_mem_status;
    wire                                active_mem_wr_back_pressure;

    lotus_active_mem #(
        .C_S_AXI_DATA_DEPTH     (C_S00_AXI_DATA_DEPTH),
        .C_S_AXI_DATA_WIDTH     (C_S00_AXI_DATA_WIDTH),
        .C_S_AXI_ADDR_WIDTH     (C_S00_AXI_ADDR_WIDTH),
        .N_STATUS_WORDS         (MEM_N_STATUS_WORDS))
    I_lotus_active_mem (
        .S_AXI_ACLK             (s00_axi_aclk),
        .S_AXI_ARESETN          (s00_axi_aresetn),

        // Upstream write interfaces
        .CH0_WR_REQ             (axi_wr_req),
        .CH0_WR_ADDR            (axi_wr_addr),
        .CH0_WR_DATA            (axi_wr_data),
        .CH0_WR_STRB            (axi_wr_strb),
        .CH0_WR_ACK             (axi_wr_ack),

        .CH1_WR_REQ             (mvr_wr_req),
        .CH1_WR_ADDR            (mvr_wr_addr),
        .CH1_WR_DATA            (mvr_wr_data),
        .CH1_WR_STRB            (mvr_wr_strb),
        .CH1_WR_ACK             (mvr_wr_ack),

        // Upstream read interfaces
        .CH0_RD_EN              (axi_rd_en),
        .CH0_RD_ADDR            (axi_rd_addr),
        .CH0_RD_VALID           (axi_rd_valid),
        .CH0_RD_DATA            (axi_rd_data),

        // Downstream interface
        .STATUS                 (active_mem_status),
        .WR_BACK_PRESSURE       (active_mem_wr_back_pressure),
        .MEMORY                 (active_mem),
        .READ_VALIDS            (),
        .WRITE_ENABLES          (active_mem_wr_enables)
    );


    // ------------------------------------------------------------------------
    // Memory map and miscellaneous control
    // ------------------------------------------------------------------------
    localparam integer STEER_CONTROL_WORD = C_S00_AXI_DATA_DEPTH - 2;
    localparam integer MOVER_CONTROL_WORD = C_S00_AXI_DATA_DEPTH - 4;
    localparam integer LASER_CONTROL_WORD = C_S00_AXI_DATA_DEPTH - 5;

    localparam integer MEM_STEER_CONFIG_OFFSET = (C_S00_AXI_DATA_DEPTH - 3) * C_S00_AXI_DATA_WIDTH;
    localparam integer MEM_STEER_CONFIG_N_BITS = 1 * C_S00_AXI_DATA_WIDTH;

    localparam integer MEM_LASER_CONFIG_OFFSET = (C_S00_AXI_DATA_DEPTH - 7) * C_S00_AXI_DATA_WIDTH;
    localparam integer MEM_LASER_CONFIG_N_BITS = 2 * C_S00_AXI_DATA_WIDTH;

    localparam integer MEM_TABLE_OFFSET = 0;
    localparam integer MEM_TABLE_N_BITS = TABLE_N_BITS;


    // unpack the memory
    wire [C_S00_AXI_DATA_WIDTH - 1 : 0] active_mem_u [C_S00_AXI_DATA_DEPTH - 1 : 0];

    genvar i_gv;
    generate
        for( i_gv = 0; i_gv < C_S00_AXI_DATA_DEPTH; i_gv = i_gv + 1 )
        begin : unpack
            assign active_mem_u[i_gv] = active_mem[C_S00_AXI_DATA_WIDTH * (i_gv+1) - 1 :
                                                   C_S00_AXI_DATA_WIDTH * i_gv];
        end
    endgenerate


    // Status
    wire driver_is_done;

    assign active_mem_status = {30'd0,
                                driver_is_done,
                                mvr_txfer_busy};


    // Config/Control: Table Mover
    always @( posedge s00_axi_aclk )
    begin
        if( s00_axi_aresetn == 1'b0 )
            mvr_start_txfer <= 1'b0;
        else
            mvr_start_txfer <= active_mem_wr_enables[MOVER_CONTROL_WORD];
    end

    assign mvr_table_idx = active_mem_u[MOVER_CONTROL_WORD][$clog2(ROM_N_TABLES) - 1 : 0];


    // Config/Control: Steering Core
    wire    [C_S00_AXI_DATA_WIDTH - 1 : 0]      steer_control;
    wire                                        steer_control_wen;
    wire    [MEM_STEER_CONFIG_N_BITS - 1 : 0]   steer_config;
    wire    [MEM_TABLE_N_BITS - 1 : 0]          steer_table;

    assign steer_control = active_mem_u[STEER_CONTROL_WORD][C_S00_AXI_DATA_WIDTH - 1 : 0];

    assign steer_control_wen = active_mem_wr_enables[STEER_CONTROL_WORD];

    assign steer_config = active_mem[MEM_STEER_CONFIG_OFFSET +: MEM_STEER_CONFIG_N_BITS];

    assign steer_table = active_mem[MEM_TABLE_OFFSET +: MEM_TABLE_N_BITS];


    // Config/Control: Laser controller
    wire    [C_S00_AXI_DATA_WIDTH - 1 : 0]      laser_control;
    reg                                         laser_control_valid;
    wire    [MEM_LASER_CONFIG_N_BITS - 1 : 0]   laser_config;

    always @( posedge s00_axi_aclk )
    begin
        if( s00_axi_aresetn == 1'b0 )
            laser_control_valid <= 1'b0;
        else
            laser_control_valid <= active_mem_wr_enables[LASER_CONTROL_WORD];
    end

    assign laser_control = active_mem_u[LASER_CONTROL_WORD][C_S00_AXI_DATA_WIDTH - 1 : 0];

    assign laser_config = active_mem[MEM_LASER_CONFIG_OFFSET +: MEM_LASER_CONFIG_N_BITS];


    // ------------------------------------------------------------------------
    // Beam Steering core
    // ------------------------------------------------------------------------
    wire                lvds_clk_int;
    wire    [5 : 0]     lvds_data_int;
    wire                pol_int;
    wire                tp1_int;
    wire                ito_clk_int;
    wire                state_eq_prog_int;

    lotus_core #(
        .CONTROL_N_BITS         (C_S00_AXI_DATA_WIDTH),
        .CONFIG_N_BITS          (MEM_STEER_CONFIG_N_BITS),
        .TABLE_N_BITS           (MEM_TABLE_N_BITS))
    I_lotus_core (
        .S_AXI_ACLK             (s00_axi_aclk),
        .S_AXI_ARESETN          (s00_axi_aresetn),

        .CLK                    (CLK_120),
        .CONTROL                (steer_control),
        .CONTROL_WEN            (steer_control_wen),
        .CONFIG                 (steer_config),
        .TABLE                  (steer_table),
        .DRIVER_IS_DONE         (driver_is_done),
        .SYNC_BUSY              (active_mem_wr_back_pressure),

        // GPO
        .ITO_CLK                (ito_clk_int),
        .STATE_EQ_PROG          (state_eq_prog_int),

        // mini-LVDS
        .LVDS_CLK               (lvds_clk_int),
        .LVDS_DATA              (lvds_data_int),
        .POL                    (pol_int),
        .TP1                    (tp1_int)
    );


    // ------------------------------------------------------------------------
    // Laser Control
    // ------------------------------------------------------------------------
    wire    laser_dr1_int;
    wire    laser_dr2_int;
    wire    laser_trigger_int;

    lotus_laser_control #(
        .CONFIG_N_BITS          (MEM_LASER_CONFIG_N_BITS),
        .CONTROL_N_BITS         (C_S00_AXI_DATA_WIDTH))
    I_lotus_laser_control(
        .S_AXI_ACLK             (s00_axi_aclk),
        .S_AXI_ARESETN          (s00_axi_aresetn),
        .CONFIG                 (laser_config),
        .CONTROL                (laser_control),
        .CONTROL_VALID          (laser_control_valid),

        .LASER_DR1              (laser_dr1_int),
        .LASER_DR2              (laser_dr2_int),
        .LASER_TRIGGER          (laser_trigger_int)
    );


    // ------------------------------------------------------------------------
    // Output buffers
    // ------------------------------------------------------------------------
    lotus_output_drivers I_lotus_output_drivers(
        .OEN                    (1'b1),

        // mini-lvds
        .LVDS_CLK               (lvds_clk_int),
        .LVDS_DATA              (lvds_data_int),
        .POL_INT                (pol_int),
        .TP1_INT                (tp1_int),

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
        .TP1                    (TP1),

        // GPO
        .ITO_CLK_INT            (ito_clk_int),
        .STATE_EQ_PROG_INT      (state_eq_prog_int),

        .ITO_CLK                (ITO_CLK),
        .STATE_EQ_PROG          (STATE_EQ_PROG),

        // Laser control
        .LASER_DR1_INT          (laser_dr1_int),
        .LASER_DR2_INT          (laser_dr2_int),
        .LASER_TRIGGER_INT      (laser_trigger_int),

        .LASER_DR1_N            (LASER_DR1_N),
        .LASER_DR1_P            (LASER_DR1_P),
        .LASER_DR2_N            (LASER_DR2_N),
        .LASER_DR2_P            (LASER_DR2_P),
        .LASER_TRIGGER          (LASER_TRIGGER)
    );


endmodule
