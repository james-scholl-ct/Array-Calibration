`include "lotus_defines.v"
`timescale 1 ns / 1 ps

module lotus_fsm_scan_seq #(
    parameter integer CNT_N_BITS = 20,                  // up to 8.7ms @ 120MHz
    parameter [CNT_N_BITS - 1 : 0] N_CYCLES_TP1 = 'd60, // 500ns @ 120MHz
    parameter [CNT_N_BITS - 1 : 0] N_CYCLES_POL = 'd2,  // > 6ns @ 120MHz
    parameter integer ITO_CNT_N_BITS = 8
)
(
    input   wire                                        APPLY,
    input   wire                                        CLK,
    input   wire    [CNT_N_BITS - 1 : 0]                DWELL_N_CYCLES,
    input   wire                                        ENABLE,
    input   wire                                        INIT,
    input   wire    [ITO_CNT_N_BITS - 1 : 0]            ITO_TC,
    input   wire                                        ITO_INVERT,
    input   wire                                        POL_OVR,
    input   wire                                        PROG_DONE,
    input   wire                                        PROG_TRIGGER_MODE,
    input   wire                                        RESET,

    output  wire                                        FORCE_ONES,
    output  reg                                         ITO_CLK,
    output  reg                                         POL,
    output  wire                                        PROG_START,
    output  reg                                         STATE_EQ_DONE,
    output  reg                                         STATE_EQ_PROG,
    output  reg                                         TP1
);


    localparam integer STATE_N_BITS = 4;
    localparam [STATE_N_BITS - 1 : 0] STATE_IDLE        = 'd0;
    localparam [STATE_N_BITS - 1 : 0] STATE_INIT_TP1_1H = 'd1;
    localparam [STATE_N_BITS - 1 : 0] STATE_INIT_TP1_1L = 'd2;
    localparam [STATE_N_BITS - 1 : 0] STATE_INIT_TP1_2H = 'd3;
    localparam [STATE_N_BITS - 1 : 0] STATE_INIT_TP1_2L = 'd4;
    localparam [STATE_N_BITS - 1 : 0] STATE_INIT_TP1_3H = 'd5;
    localparam [STATE_N_BITS - 1 : 0] STATE_INIT_PROG   = 'd6;
    localparam [STATE_N_BITS - 1 : 0] STATE_DONE        = 'd7;
    localparam [STATE_N_BITS - 1 : 0] STATE_PROG        = 'd8;
    localparam [STATE_N_BITS - 1 : 0] STATE_WAIT        = 'd9;
    localparam [STATE_N_BITS - 1 : 0] STATE_POL         = 'd10;
    localparam [STATE_N_BITS - 1 : 0] STATE_TP1         = 'd11;
    localparam [STATE_N_BITS - 1 : 0] STATE_PROG_FIN    = 'd12;
    localparam [STATE_N_BITS - 1 : 0] STATE_WAIT_FIN    = 'd13;
    localparam [STATE_N_BITS - 1 : 0] STATE_POL_FIN     = 'd14;
    localparam [STATE_N_BITS - 1 : 0] STATE_TP1_FIN     = 'd15;


    reg  [CNT_N_BITS - 1 : 0]   cnt;
    wire                        skip_reset;
    wire                        reset_cnt;
    wire                        pol_cnt_done;
    wire                        tp1_cnt_done;
    wire                        dwell_cnt_done;
    reg                         dwell_expired;
    reg  [STATE_N_BITS - 1 : 0] state;
    reg  [STATE_N_BITS - 1 : 0] state_d;
    wire                        clear_apply;
    reg                         apply_is_pending;
    reg                         phase;

    // Allow counter to continue running through PROG and WAIT (i.e. count
    // from TP1 start to POL start for dwell).
    assign skip_reset = (state_d == STATE_PROG) || (state_d == STATE_PROG_FIN) ||
                        (state_d == STATE_WAIT) || (state_d == STATE_WAIT_FIN);

    assign reset_cnt = (state == STATE_IDLE) ||
                       (state == STATE_DONE) ||
                       (state != state_d) && !skip_reset;

    assign pol_cnt_done = (cnt == N_CYCLES_POL - 1);

    assign tp1_cnt_done = (cnt == N_CYCLES_TP1 - 1);

    assign dwell_cnt_done = (cnt == DWELL_N_CYCLES);

    always @( posedge CLK )
    begin
        if( RESET )
        begin
            cnt <= 'd0;
            dwell_expired <= 1'b0;
        end
        else
        begin
            if( reset_cnt )
            begin
                cnt <= 'd0;
                dwell_expired <= 1'b0;
            end
            else
            begin
                cnt <= cnt + 'd1;
                // stick this high in case user sets dwell_cnt too low (fast)
                dwell_expired <= dwell_expired || dwell_cnt_done;
            end
        end
    end

    assign clear_apply = (state != state_d) && (state_d == STATE_PROG ||
                                                state_d == STATE_PROG_FIN);

    reg prog_occurred;
    always @( posedge CLK )
    begin
        if( RESET )
        begin
            state <= STATE_IDLE;
            STATE_EQ_DONE <= 1'b0;
            prog_occurred <= 1'b0;
            STATE_EQ_PROG <= 1'b0;
            apply_is_pending <= 1'b0;
        end
        else
        begin
            state <= state_d;
            STATE_EQ_DONE <= (state == STATE_DONE);

            // evaluate once for each normal TP1 pulse
            if (state == STATE_TP1 && !TP1)
            begin
                // Two modes per PROG_TRIGGER_MODE: 1) pulse and 0) toggle.
                //  - 1: On TP1, pulse high if prog occurred else set TP1 low
                //  - 0: On TP1, toggle STATE_EQ_PROG if prog occurred
                STATE_EQ_PROG <= PROG_TRIGGER_MODE ? prog_occurred :
                                 prog_occurred ^ STATE_EQ_PROG;
                prog_occurred <= 1'b0;
            end
            else if(state == STATE_PROG)
            begin
                prog_occurred <= 1'b1;
            end

            if( clear_apply )
                apply_is_pending <= 1'b0;
            else if( APPLY )
                apply_is_pending <= 1'b1;
        end
    end


    always @(*)
    begin
        case( state )
            STATE_IDLE:
            begin
                state_d = INIT ? STATE_INIT_TP1_1H : state;
            end

            STATE_INIT_TP1_1H:
            begin
                state_d = tp1_cnt_done ? STATE_INIT_TP1_1L : state;
            end

            STATE_INIT_TP1_1L:
            begin
                state_d = tp1_cnt_done ? STATE_INIT_TP1_2H : state;
            end

            STATE_INIT_TP1_2H:
            begin
                state_d = tp1_cnt_done ? STATE_INIT_TP1_2L : state;
            end

            STATE_INIT_TP1_2L:
            begin
                state_d = tp1_cnt_done ? STATE_INIT_TP1_3H : state;
            end

            STATE_INIT_TP1_3H:
            begin
                state_d = tp1_cnt_done ? STATE_INIT_PROG : state;
            end

            STATE_INIT_PROG:
            begin
                state_d = PROG_DONE ? STATE_DONE : state;
            end

            STATE_DONE:
            begin
                state_d = (ENABLE && apply_is_pending) ? STATE_PROG : state;
            end

            STATE_PROG:
            begin
                state_d = PROG_DONE ? STATE_WAIT : state;
            end

            STATE_WAIT:
            begin
                state_d = dwell_expired ? STATE_POL : state;
            end

            STATE_POL:
            begin
                state_d = pol_cnt_done ? STATE_TP1 : state;
            end

            STATE_TP1:
            begin
                if( tp1_cnt_done )
                begin
                    if( phase )
                    begin
                        // need to do second phase of polarity
                        state_d = STATE_WAIT;
                    end
                    else if( !phase && !ENABLE )
                    begin
                        // user termination
                        state_d = STATE_PROG_FIN;
                    end
                    else if( !phase && ENABLE && apply_is_pending )
                    begin
                        // process request to program next angle
                        state_d = STATE_PROG;
                    end
                    else //if( !phase && ENABLE && !apply_is_pending )
                    begin
                        // no request to reprogram
                        state_d = STATE_WAIT;
                    end
                end
                else
                begin
                    state_d = state;
                end
            end

            STATE_PROG_FIN:
            begin
                state_d = PROG_DONE ? STATE_WAIT_FIN : state;
            end

            STATE_WAIT_FIN:
            begin
                state_d = dwell_expired ? STATE_POL_FIN : state;
            end

            STATE_POL_FIN:
            begin
                state_d = pol_cnt_done ? STATE_TP1_FIN : state;
            end

            STATE_TP1_FIN:
            begin
                if( tp1_cnt_done )
                begin
                    if( phase )
                    begin
                        // need to do second phase of polarity
                        state_d = STATE_WAIT_FIN;
                    end
                    else
                    begin
                        // done with secnd phase
                        state_d = STATE_DONE;
                    end
                end
                else
                begin
                    state_d = state;
                end
            end

            default:
            begin
                state_d = state;
            end
        endcase
    end


    // Outputs
    assign FORCE_ONES = (state == STATE_INIT_PROG) ||
                        (state == STATE_PROG_FIN);


    always @( posedge CLK )
    begin
        if( RESET )
        begin
            POL <= 1'b0;
            phase <= 1'b0;
        end
        else if( state == STATE_POL && cnt == 'd0 )
        begin
            // only update once per POL state
            POL <= POL_OVR ? 1'b0 : ~POL;
            phase <= ~phase;
        end
        else if( state == STATE_POL_FIN && cnt == 'd0 )
        begin
            // only update once per POL state
            // POL only deviates from phase when finishing such that POL=0
            // and coefficients of all-ones causes 0V on all (odd) channels.
            POL <= 1'b0;
            phase <= ~phase;
        end
    end


    assign PROG_START = (state == STATE_INIT_PROG) ||
                        (state == STATE_PROG)      ||
                        (state == STATE_PROG_FIN);


    reg  [ITO_CNT_N_BITS - 1 : 0]   ito_cnt;

    always @( posedge CLK )
    begin
        if( RESET )
        begin
            TP1 <= 1'b0;
            ito_cnt <= 'd0;
            ITO_CLK <= 1'b0;
        end
        else
        begin
            // hold TP1 high during DONE to make driver outputs float
            TP1 <= (state == STATE_INIT_TP1_1H ) ||
                   (state == STATE_INIT_TP1_2H ) ||
                   (state == STATE_INIT_TP1_3H ) ||
                   (state == STATE_TP1         ) ||
                   (state == STATE_TP1_FIN     ) ||
                   (state == STATE_DONE        );

            // evaluate once for each normal TP1 pulse
            if (state == STATE_TP1 && !TP1)
            begin
                if (ito_cnt == ITO_TC)
                begin
                    ito_cnt <= 'd0;
                    // ITO_TC == 0 indicates the mode where POL/phase and
                    // ITO_CLK are at the same frequency and the notion of
                    // inverting ITO_CLK relative to POL/phase has meaning.
                    //
                    // ITO_TC != 0 indicates that ITO_CLK will toggle
                    // after the specified number of TP1 pulses.
                    ITO_CLK <= (ITO_TC == 'd0) ? (phase ^ ITO_INVERT) :
                                                 ~ITO_CLK;
                end
                else
                begin
                    ito_cnt <= ito_cnt + 'd1;
                    ITO_CLK <= ITO_CLK;
                end
            end
            else if (state == STATE_DONE || state == STATE_IDLE)
            begin
                ito_cnt <= 'd0;
                ITO_CLK <= 1'b0;
            end
        end
    end

endmodule
