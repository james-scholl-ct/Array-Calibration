
`timescale 1 ns / 1 ps

module orchid_core #(
    // Depth of the AXI RAM in words of C_S_AXI_DATA_WIDTH bits
    parameter integer C_S_AXI_DATA_DEPTH = 72,
    // Width of AXI data bus
    parameter integer C_S_AXI_DATA_WIDTH = 32
)
(
    input   wire                                CLK,
    input   wire    [C_S_AXI_DATA_DEPTH*C_S_AXI_DATA_WIDTH-1 : 0 ] AXI_MEMORY,
    input   wire    [C_S_AXI_DATA_WIDTH-1 : 0]  CONTROL,
    input   wire                                CONTROL_VALID,

    output  wire                                LVDS_0N,
    output  wire                                LVDS_0P,
    output  wire                                LVDS_1N,
    output  wire                                LVDS_1P,
    output  wire                                LVDS_2N,
    output  wire                                LVDS_2P,
    output  wire                                LVDS_3N,
    output  wire                                LVDS_3P,
    output  wire                                LVDS_4N,
    output  wire                                LVDS_4P,
    output  wire                                LVDS_5N,
    output  wire                                LVDS_5P,
    output  wire                                LVDS_CLK_N,
    output  wire                                LVDS_CLK_P,
    output  wire                                POL,
    output  wire                                TP1
);


    localparam integer DWELL_CNT_N_BITS = 20;


    // Synchronized control
    //      CONTROL_VALID strobes with each update to the CONTROL register
    wire            reset;
    wire            start;
    wire [ 1 : 0 ]  mode;

    assign reset = CONTROL[0] & CONTROL_VALID;
    assign start = CONTROL[1];
    assign mode  = CONTROL[3:2];


    // Memory map (NOT SYNCHRONIZED)
    wire [C_S_AXI_DATA_WIDTH - 1 : 0] axi_memory_u [C_S_AXI_DATA_DEPTH - 1 : 0];
    wire [DWELL_CNT_N_BITS - 1 : 0] dwell_n_cycles_normal;
    wire [DWELL_CNT_N_BITS - 1 : 0] dwell_n_cycles_mode00;

    genvar i_gv;
    generate
        for( i_gv = 0; i_gv < C_S_AXI_DATA_DEPTH; i_gv = i_gv + 1 )
        begin : array_reshape
            assign axi_memory_u[ i_gv ] =
                AXI_MEMORY[ C_S_AXI_DATA_WIDTH * (i_gv+1) - 1 : C_S_AXI_DATA_WIDTH * i_gv ];
        end
    endgenerate

    assign dwell_n_cycles_normal = axi_memory_u[C_S_AXI_DATA_DEPTH - 3][DWELL_CNT_N_BITS - 1 : 0];
    assign dwell_n_cycles_mode00 = axi_memory_u[C_S_AXI_DATA_DEPTH - 4][DWELL_CNT_N_BITS - 1 : 0];


    // High-level state machine to sequence the angles
    wire            prog_done;
    wire            force_zeros;
    wire            pol_pre;
    wire            prog_start;
    wire [ 1 : 0 ]  table_sel;
    wire            tp1_pre;

    orchid_fsm_scan_seq #(
        .CNT_N_BITS             (DWELL_CNT_N_BITS))
    I_orchid_fsm_scan_seq (
        .CLK                    (CLK),
        .DWELL_N_CYCLES_MODE00  (dwell_n_cycles_mode00),
        .DWELL_N_CYCLES_NORMAL  (dwell_n_cycles_normal),
        .MODE                   (mode),
        .PROG_DONE              (prog_done),
        .RESET                  (reset),
        .START                  (start),

        .FORCE_ZEROS            (force_zeros),
        .POL                    (pol_pre),
        .PROG_START             (prog_start),
        .TABLE_SEL              (table_sel),
        .TP1                    (tp1_pre)
    );


    // State machine to sequence the mini-lvds data transfer
    wire [ 2 : 0 ]  bit_sel;
    wire            lvds_clk_pre;
    wire            rst_en;
    wire [ 7 : 0 ]  step;
    wire            tx_en;

    orchid_fsm_lvds_data I_orchid_fsm_lvds_data (
        .CLK            (CLK),
        .RESET          (reset),
        .START          (prog_start),

        .BIT_SEL        (bit_sel),
        .DONE           (prog_done),
        .LVDS_CLK       (lvds_clk_pre),
        .RST_EN         (rst_en),
        .STEP           (step),
        .TX_EN          (tx_en)
    );


    // Datapath: coefficient mux
    wire [ 5 : 0 ]  lvds_data;

    orchid_data_mux #(
        .C_S_AXI_DATA_DEPTH     (C_S_AXI_DATA_DEPTH),
        .C_S_AXI_DATA_WIDTH     (C_S_AXI_DATA_WIDTH),
        .N_DRIVE_CHANNELS       (64))
    I_orchid_data_mux (
        .AXI_MEMORY             (AXI_MEMORY),
        .BIT_SEL                (bit_sel),
        .CLK                    (CLK),
        .FORCE_ZEROS            (force_zeros),
        .RESET                  (reset),
        .RST_EN                 (rst_en),
        .STEP                   (step),
        .TABLE_SEL              (table_sel),
        .TX_EN                  (tx_en),

        .LVDS_DATA              (lvds_data)
    );


    // Output buffers
    orchid_output_drivers I_orchid_output_drivers(
        .LVDS_0         (lvds_data[0]),
        .LVDS_1         (lvds_data[1]),
        .LVDS_2         (lvds_data[2]),
        .LVDS_3         (lvds_data[3]),
        .LVDS_4         (lvds_data[4]),
        .LVDS_5         (lvds_data[5]),
        .LVDS_CLK       (lvds_clk_pre),
        .OEN            (1'b1),
        .POL_INT        (pol_pre),
        .TP1_INT        (tp1_pre),

        .LVDS_0N        (LVDS_0N),
        .LVDS_0P        (LVDS_0P),
        .LVDS_1N        (LVDS_1N),
        .LVDS_1P        (LVDS_1P),
        .LVDS_2N        (LVDS_2N),
        .LVDS_2P        (LVDS_2P),
        .LVDS_3N        (LVDS_3N),
        .LVDS_3P        (LVDS_3P),
        .LVDS_4N        (LVDS_4N),
        .LVDS_4P        (LVDS_4P),
        .LVDS_5N        (LVDS_5N),
        .LVDS_5P        (LVDS_5P),
        .LVDS_CLK_N     (LVDS_CLK_N),
        .LVDS_CLK_P     (LVDS_CLK_P),
        .POL            (POL),
        .TP1            (TP1)
    );

endmodule
