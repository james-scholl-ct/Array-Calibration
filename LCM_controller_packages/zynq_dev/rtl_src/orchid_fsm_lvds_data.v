`timescale 1 ns / 1 ps

module orchid_fsm_lvds_data #(
    parameter CNT_N_BITS = 'd4,
    parameter N_CYCLES_RESET_RST = 'd10,
    parameter N_CYCLES_TX_WAIT = 'd14
)
(
    input                               CLK,
    input                               RESET,
    input                               START,

    output      [ 2 : 0 ]               BIT_SEL,
    output                              DONE,
    output reg                          LVDS_CLK,
    output                              RST_EN,
    output reg  [ 7 : 0 ]               STEP,
    output                              TX_EN
);


    // 171 steps with each LVDS channel sending 8 bits per step.
    localparam [ 7 : 0 ]              LAST_STEP = 'd170;
    localparam [ CNT_N_BITS - 1 : 0 ] LAST_LVDS_BIT = 'd7;

    localparam [ 2 : 0 ] STATE_IDLE       = 'd0;
    localparam [ 2 : 0 ] STATE_RST        = 'd1;
    localparam [ 2 : 0 ] STATE_RST_WAIT   = 'd2;
    localparam [ 2 : 0 ] STATE_TX         = 'd3;
    localparam [ 2 : 0 ] STATE_TX_WAIT    = 'd4;
    localparam [ 2 : 0 ] STATE_DONE       = 'd5;


    // lvds clock (on negedge)
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


    reg  [ CNT_N_BITS - 1 : 0 ] cnt;
    reg  [ 2 : 0 ]              state;
    reg  [ 2 : 0 ]              state_d;

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
                state_d = ( cnt == N_CYCLES_RESET_RST - 1 ) ? STATE_RST_WAIT : state;
            end

            STATE_RST_WAIT:
            begin
                // the driver is sensitive to the phase of the LVDS clock:
                // RST is sampled on posedge CLK+ and first data is sampled
                // on the posedge of CLK+ following a detection of RST low.
                state_d = ( lvds_clk_phase && cnt != 'd0 ) ? STATE_TX : state;
            end

            STATE_TX:
            begin
                state_d = ( STEP == LAST_STEP && cnt == LAST_LVDS_BIT ) ? STATE_TX_WAIT : state;
            end

            STATE_TX_WAIT:
            begin
                // slop + 5 cycles of LVDS CLK+ (10 cycles of core CLK)
                state_d = ( cnt == N_CYCLES_TX_WAIT - 1 ) ? STATE_DONE : state;
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
        end
        else
        begin
            if( state != state_d )
            begin
                cnt <= 'd0;
            end
            else if( state == STATE_TX && cnt == LAST_LVDS_BIT )
            begin
                // reset bit counter for each step
                cnt <= 'd0;
            end
            else if( state != STATE_IDLE && state != STATE_DONE )
            begin
                // increment when not in idle or done
                cnt <= cnt + 'd1;
            end
        end
    end


    always @( posedge CLK )
    begin
        if( RESET )
        begin
            STEP <= 'd0;
        end
        else
        begin
            if( state != STATE_TX )
                STEP <= 'd0;
            else
                STEP <= ( cnt == LAST_LVDS_BIT ) ? STEP + 'd1 : STEP;
        end
    end


    assign BIT_SEL = ( state == STATE_TX ) ? cnt[ 2 : 0 ] : 'd0;

    assign DONE = ( state == STATE_DONE );

    assign RST_EN = ( state == STATE_RST );

    assign TX_EN = ( state == STATE_TX );

endmodule
