`include "lotus_defines.v"
`timescale 1 ns / 1 ps

module lotus_core #(
    parameter integer CONTROL_N_BITS = 32,
    parameter integer CONFIG_N_BITS = 32,
    parameter integer TABLE_N_BITS = 32
)
(
    input   wire                                        S_AXI_ACLK,
    input   wire                                        S_AXI_ARESETN,

    input   wire                                        CLK,
    input   wire    [CONTROL_N_BITS - 1 : 0]            CONTROL,
    input   wire                                        CONTROL_WEN,
    input   wire    [CONFIG_N_BITS - 1 : 0]             CONFIG,
    input   wire    [TABLE_N_BITS - 1 : 0]              TABLE,
    output  wire                                        DRIVER_IS_DONE,
    output  wire                                        SYNC_BUSY,

    // GPO
    output  wire                                        ITO_CLK,
    output  wire                                        STATE_EQ_PROG,

    // mini-lvds
    output  wire                                        LVDS_CLK,
    output  wire    [5 : 0]                             LVDS_DATA,
    output  wire                                        POL,
    output  wire                                        TP1
);


    localparam integer DWELL_CNT_N_BITS = 20;
    localparam integer ITO_CNT_N_BITS = 8;

    localparam integer CONTROL_RESET_IDX        = 0;
    localparam integer CONTROL_INIT_IDX         = 1;
    localparam integer CONTROL_ENABLE_IDX       = 2;
    localparam integer CONTROL_APPLY_IDX        = 3;
    localparam integer CONTROL_POL_OVERRIDE_IDX = 4;


    // ------------------------------------------------------------------------
    // CDC and Buffering
    // ------------------------------------------------------------------------
    wire    [CONTROL_N_BITS - 1 : 0]    control_syncd;
    wire                                control_syncd_dv;
    wire    [TABLE_N_BITS - 1 : 0]      table_syncd;
    wire                                reset;
    wire                                init;
    wire                                enable;
    wire                                apply;
    wire                                pol_ovr;

    // The table is double-buffered (might as well synchronize it too). This
    // CDC guarantees stable data by applying back-pressure to the active_mem
    // as soon as active_mem asserts the write enable of the CONTROL word.
    // Back-pressure is not removed until both A_DATA and A_DATA2 have been
    // transferred (i.e. after writing CONTROL, *all* writes are blocked for
    // a few cycles).
    lotus_sync_a2b #(
        .N_DATA_BITS            (CONTROL_N_BITS),
        .N_DATA2_BITS           (TABLE_N_BITS))
    I_lotus_sync_a2b (
        .A_CLK                  (S_AXI_ACLK),
        .A_DATA                 (CONTROL),
        .A_DATA2                (TABLE),
        .A_REQ                  (CONTROL_WEN),
        .A_SRESET               (~S_AXI_ARESETN),
        .B_CLK                  (CLK),

        .A_BUSY                 (SYNC_BUSY),
        .B_DATA                 (control_syncd),
        .B_DATA2                (table_syncd),
        .B_DATA_VALID           (control_syncd_dv)
    );

    assign reset   = control_syncd[CONTROL_RESET_IDX] & control_syncd_dv;
    assign init    = control_syncd[CONTROL_INIT_IDX] & control_syncd_dv;
    assign enable  = control_syncd[CONTROL_ENABLE_IDX];
    assign apply   = control_syncd[CONTROL_APPLY_IDX] & control_syncd_dv;
    assign pol_ovr = control_syncd[CONTROL_POL_OVERRIDE_IDX];

    // Status
    wire state_eq_done;
    reg  state_eq_done_sync0;
    reg  state_eq_done_sync1;

    always @( posedge S_AXI_ACLK )
    begin
        if ( ~S_AXI_ARESETN )
        begin
            state_eq_done_sync0 <= 1'b0;
            state_eq_done_sync1 <= 1'b0;
        end
        else
        begin
            state_eq_done_sync0 <= state_eq_done;
            state_eq_done_sync1 <= state_eq_done_sync0;
        end
    end

    assign DRIVER_IS_DONE = state_eq_done_sync1;


    // ------------------------------------------------------------------------
    // Config
    //      (CONFIG IS NOT SYNCHRONIZED--IT SHOULDN'T CHANGE WHILE RUNNING)
    // ------------------------------------------------------------------------
    wire [DWELL_CNT_N_BITS - 1 : 0] dwell_n_cycles;
    wire [ITO_CNT_N_BITS - 1 : 0]   ito_tc;
    wire                            ito_invert;
    wire                            prog_trigger_mode;

    assign dwell_n_cycles = CONFIG[DWELL_CNT_N_BITS - 1 : 0];

    assign ito_tc = CONFIG[DWELL_CNT_N_BITS +: ITO_CNT_N_BITS];

    assign ito_invert = CONFIG[DWELL_CNT_N_BITS + ITO_CNT_N_BITS];

    assign prog_trigger_mode = CONFIG[DWELL_CNT_N_BITS + ITO_CNT_N_BITS + 1];

    // ------------------------------------------------------------------------
    // Mini-LVDS
    // ------------------------------------------------------------------------
    wire            force_ones;
    wire            prog_start;
    wire            prog_done;

    lotus_fsm_scan_seq #(
        .CNT_N_BITS             (DWELL_CNT_N_BITS),
        .ITO_CNT_N_BITS         (ITO_CNT_N_BITS))
    I_lotus_fsm_scan_seq (
        .APPLY                  (apply),
        .CLK                    (CLK),
        .DWELL_N_CYCLES         (dwell_n_cycles),
        .ENABLE                 (enable),
        .INIT                   (init),
        .ITO_TC                 (ito_tc),
        .ITO_INVERT             (ito_invert),
        .POL_OVR                (pol_ovr),
        .PROG_DONE              (prog_done),
        .PROG_TRIGGER_MODE      (prog_trigger_mode),
        .RESET                  (reset),

        .FORCE_ONES             (force_ones),
        .ITO_CLK                (ITO_CLK),
        .POL                    (POL),
        .PROG_START             (prog_start),
        .STATE_EQ_DONE          (state_eq_done),
        .STATE_EQ_PROG          (STATE_EQ_PROG),
        .TP1                    (TP1)
    );


    lotus_fsm_lvds_data #(
        .N_STEPS                (152),
        .TABLE_N_BITS           (TABLE_N_BITS))
    I_lotus_fsm_lvds_data (
        .CLK                    (CLK),
        .FORCE_ONES             (force_ones),
        .RESET                  (reset),
        .START                  (prog_start),
        .TABLE                  (table_syncd),

        .DONE                   (prog_done),
        .LVDS_CLK               (LVDS_CLK),
        .LVDS_DATA              (LVDS_DATA)
    );

endmodule
