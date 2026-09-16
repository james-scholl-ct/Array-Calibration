`include "lotus_defines.v"
`timescale 1 ns / 1 ps

module lotus_fsm_lvds_data #(
    parameter integer N_STEPS = 152,
    parameter integer TABLE_N_BITS = 32
)
(
    input   wire                                        CLK,
    input   wire                                        FORCE_ONES,
    input   wire                                        RESET,
    input   wire                                        START,
    input   wire    [TABLE_N_BITS - 1 : 0]              TABLE,

    output  wire                                        DONE,
    output  reg                                         LVDS_CLK,
    output  reg     [5 : 0]                             LVDS_DATA
);


    // N_STEPS steps with each LVDS channel sending 8 bits per step.
    localparam integer STEP_N_BITS = $clog2(N_STEPS);
    localparam [STEP_N_BITS - 1 : 0] LAST_STEP = N_STEPS - 1;

    localparam integer CNT_N_BITS = 'd4;
    localparam [CNT_N_BITS - 1 : 0] N_CYCLES_RESET_RST = 'd10;
    localparam [CNT_N_BITS - 1 : 0] N_CYCLES_TX_WAIT = 'd14;
    localparam [CNT_N_BITS - 1 : 0] LAST_STEP_BIT = 'd7;

    localparam STATE_N_BITS = 'd3;
    localparam [STATE_N_BITS - 1 : 0] STATE_IDLE     = 'd0;
    localparam [STATE_N_BITS - 1 : 0] STATE_RST      = 'd1;
    localparam [STATE_N_BITS - 1 : 0] STATE_RST_WAIT = 'd2;
    localparam [STATE_N_BITS - 1 : 0] STATE_TX       = 'd3;
    localparam [STATE_N_BITS - 1 : 0] STATE_TX_WAIT  = 'd4;
    localparam [STATE_N_BITS - 1 : 0] STATE_DONE     = 'd5;


    // ------------------------------------------------------------------------
    // lvds clock (on negedge)
    // ------------------------------------------------------------------------
    reg lvds_clk_phase;

    always @( posedge CLK )
    begin
        if( RESET )
        begin
            lvds_clk_phase <= 1'b0;
        end
        else
        begin
            lvds_clk_phase <= ~lvds_clk_phase;
        end
    end


    always @( negedge CLK )
    begin
        if( RESET )
        begin
            LVDS_CLK <= 1'b0;
        end
        else
        begin
            LVDS_CLK <= lvds_clk_phase;
        end
    end


    // ------------------------------------------------------------------------
    // Control
    // ------------------------------------------------------------------------
    reg  [STATE_N_BITS - 1 : 0] state;
    reg  [STATE_N_BITS - 1 : 0] state_d;
    reg  [CNT_N_BITS - 1 : 0]   cnt;
    reg  [STEP_N_BITS - 1 : 0]  step;

    always @(*)
    begin
        case( state )
            STATE_IDLE:
            begin
                state_d = START ? STATE_RST : state;
            end

            STATE_RST:
            begin
                // 5 cycles of LVDS CLK+ (10 cycles of core CLK)
                state_d = (cnt == N_CYCLES_RESET_RST - 1) ? STATE_RST_WAIT : state;
            end

            STATE_RST_WAIT:
            begin
                // the driver is sensitive to the phase of the LVDS clock:
                // RST is sampled on posedge CLK+ and first data is sampled
                // on the posedge of CLK+ following a detection of RST low.
                state_d = (lvds_clk_phase && cnt != 'd0) ? STATE_TX : state;
            end

            STATE_TX:
            begin
                state_d = (step == LAST_STEP && cnt == LAST_STEP_BIT) ? STATE_TX_WAIT : state;
            end

            STATE_TX_WAIT:
            begin
                // slop + 5 cycles of LVDS CLK+ (10 cycles of core CLK)
                state_d = (cnt == N_CYCLES_TX_WAIT - 1) ? STATE_DONE : state;
            end

            STATE_DONE:
            begin
                state_d = !START ? STATE_IDLE : state;
            end

            default:
            begin
                state_d = STATE_IDLE;
            end
        endcase
    end


    always @( posedge CLK )
    begin
        if( RESET )
        begin
            state <= STATE_IDLE;
        end
        else
        begin
            state <= state_d;
        end
    end


    always @( posedge CLK )
    begin
        if( RESET )
        begin
            cnt <= 'd0;
            step <= 'd0;
        end
        else
        begin
            if( state != state_d || state == STATE_IDLE || state == STATE_DONE )
            begin
                cnt <= 'd0;
                step <= 'd0;
            end
            else if( state == STATE_TX && cnt == LAST_STEP_BIT )
            begin
                cnt <= 'd0;
                step <= step + 'd1;
            end
            else
            begin
                cnt <= cnt + 'd1;
                step <= step;
            end
        end
    end

    assign DONE = (state == STATE_DONE);

    // ------------------------------------------------------------------------
    // Data
    // ------------------------------------------------------------------------
    wire [2 : 0]    bit_sel;
    wire [5 : 0]    lvds_data_muxed;

    assign bit_sel = (state == STATE_TX) ? cnt[2 : 0] : 'd0;

    lotus_data_mux #(
        .STEP_N_BITS            (STEP_N_BITS),
        .TABLE_N_BITS           (TABLE_N_BITS))
    I_lotus_data_mux (
        .BIT_SEL                (bit_sel),
        .STEP                   (step),
        .TABLE                  (TABLE),

        .LVDS_DATA              (lvds_data_muxed)
    );


    always @( posedge CLK )
    begin
        if( RESET )
        begin
            LVDS_DATA <= 6'd0;
        end
        else
        begin
            if( state == STATE_RST )
            begin
                // bit 0 is used for RST (other bits are don't cares)
                LVDS_DATA <= 6'b00000_1;
            end
            else if( state == STATE_TX )
            begin
                LVDS_DATA <= FORCE_ONES ? 6'h3f : lvds_data_muxed;
            end
            else
            begin
                LVDS_DATA <= 'd0;
            end
        end
    end

endmodule
