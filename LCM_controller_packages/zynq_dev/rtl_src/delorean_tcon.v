`include "delorean_defines.v"
`timescale 1 ns / 1 ps

module delorean_tcon #(
    parameter integer ITO_CNT_N_BITS = 8,
    parameter integer N_BUFFERS = 2,
    parameter integer TP1_PERIOD_N_BITS = 24,
    parameter integer TP1_PW_N_BITS = 8
)
(
    input   wire    [N_BUFFERS - 1 : 0]                 APPLY,
    input   wire                                        CLK,
    input   wire                                        ENABLE,
    input   wire                                        ITO_ASYNC,
    input   wire                                        ITO_INVERT,
    input   wire    [ITO_CNT_N_BITS - 1 : 0]            ITO_TC,
    input   wire                                        POL_FINISH_OVR,
    input   wire                                        POL_OVR_EN,
    input   wire                                        POL_OVR_VAL,
    input   wire                                        PROG_TRIGGER_MODE,
    input   wire                                        PROG_DONE,
    input   wire                                        RESET,
    input   wire                                        TP1_DONE_HIGH,
    input   wire    [TP1_PERIOD_N_BITS - 1 : 0]         TP1_PERIOD,
    input   wire    [TP1_PW_N_BITS - 1 : 0]             TP1_PW,

    output  reg     [N_BUFFERS - 1 : 0]                 APPLY_CACHE,
    output  reg     [$clog2(N_BUFFERS) - 1 : 0]         BUF_IDX,
    output  reg                                         ITO_CLK,
    output  reg     [N_BUFFERS - 1 : 0]                 LOADING,
    output  reg                                         POL,
    output  wire                                        PROG_START,
    output  reg                                         PROG_TRIGGER,
    output  wire    [31 : 0]                            STATE_ONEHOT,
    output  reg                                         TP1,
    output  wire                                        USE_RESET_CODE
);


    localparam integer CNT_N_BITS = TP1_PERIOD_N_BITS;
    localparam [CNT_N_BITS - 1 : 0] N_CYCLES_POL = 'd2;  // > 6ns @ 100MHz

    localparam integer STATE_N_BITS = 5;
    localparam [STATE_N_BITS - 1 : 0] STATE_IDLE        = 'd0;
    localparam [STATE_N_BITS - 1 : 0] STATE_INIT_TP1_1H = 'd1;
    localparam [STATE_N_BITS - 1 : 0] STATE_INIT_TP1_1L = 'd2;
    localparam [STATE_N_BITS - 1 : 0] STATE_INIT_TP1_2H = 'd3;
    localparam [STATE_N_BITS - 1 : 0] STATE_INIT_TP1_2L = 'd4;
    localparam [STATE_N_BITS - 1 : 0] STATE_INIT_TP1_3H = 'd5;
    localparam [STATE_N_BITS - 1 : 0] STATE_INIT_PROG   = 'd6;
    localparam [STATE_N_BITS - 1 : 0] STATE_INIT_TP1_4H = 'd7;
    localparam [STATE_N_BITS - 1 : 0] STATE_DONE        = 'd8;
    localparam [STATE_N_BITS - 1 : 0] STATE_PROG        = 'd9;
    localparam [STATE_N_BITS - 1 : 0] STATE_WAIT        = 'd10;
    localparam [STATE_N_BITS - 1 : 0] STATE_POL         = 'd11;
    localparam [STATE_N_BITS - 1 : 0] STATE_TP1         = 'd12;
    localparam [STATE_N_BITS - 1 : 0] STATE_PROG_FIN    = 'd13;
    localparam [STATE_N_BITS - 1 : 0] STATE_WAIT_FIN    = 'd14;
    localparam [STATE_N_BITS - 1 : 0] STATE_POL_FIN     = 'd15;
    localparam [STATE_N_BITS - 1 : 0] STATE_TP1_FIN     = 'd16;


    reg  [CNT_N_BITS - 1 : 0]   cnt;
    wire                        skip_reset;
    wire                        reset_cnt;
    wire                        pol_done;
    wire                        tp1_pw_done;
    wire                        tp1_period_done;
    reg                         tp1_period_expired;
    reg  [STATE_N_BITS - 1 : 0] state;
    reg  [STATE_N_BITS - 1 : 0] state_d;
    wire                        clear_apply;
    wire                        apply_is_pending;
    reg                         phase;

    // Allow counter to continue running through PROG and WAIT (i.e. count
    // from TP1 start to POL start for tp1_period).
    assign skip_reset = (state_d == STATE_PROG) || (state_d == STATE_PROG_FIN) ||
                        (state_d == STATE_WAIT) || (state_d == STATE_WAIT_FIN);

    assign reset_cnt = (state == STATE_IDLE) ||
                       (state == STATE_DONE) ||
                       (state != state_d) && !skip_reset;

    assign pol_done = (cnt == N_CYCLES_POL - 1);

    assign tp1_pw_done = (cnt == TP1_PW);

    assign tp1_period_done = (cnt == TP1_PERIOD);

    always @( posedge CLK )
    begin
        if( RESET )
        begin
            cnt <= 'd0;
            tp1_period_expired <= 1'b0;
        end
        else
        begin
            if( reset_cnt )
            begin
                cnt <= 'd0;
                tp1_period_expired <= 1'b0;
            end
            else
            begin
                cnt <= cnt + 'd1;
                // stick this high in case user sets tp1_period too low (fast)
                tp1_period_expired <= tp1_period_expired || tp1_period_done;
            end
        end
    end

    // include PROG_FIN so that APPLY_CACHE clears before reaching DONE
    assign clear_apply = (state != state_d) && (state_d == STATE_PROG ||
                                                state_d == STATE_PROG_FIN);

    assign apply_is_pending = |APPLY_CACHE;

    reg prog_occurred;
    always @( posedge CLK )
    begin
        if( RESET )
        begin
            state <= STATE_IDLE;
            PROG_TRIGGER <= 1'b0;
            prog_occurred <= 1'b0;
            APPLY_CACHE <= 'd0;
            BUF_IDX <= 'd0;
            LOADING <= 'd0;
        end
        else
        begin
            state <= state_d;

            // evaluate once for each normal TP1 pulse
            if (state == STATE_TP1 && !TP1)
            begin
                // Two modes per PROG_TRIGGER_MODE: 1) pulse and 0) toggle.
                //  - 1: On TP1, pulse high if prog occurred else set TP1 low
                //  - 0: On TP1, toggle STATE_EQ_PROG if prog occurred
                PROG_TRIGGER <= PROG_TRIGGER_MODE ? prog_occurred :
                                prog_occurred ^ PROG_TRIGGER;
                prog_occurred <= 1'b0;
            end
            else if(state == STATE_PROG)
            begin
                prog_occurred <= 1'b1;
            end

            // cache apply requests since the controller only evaluates them
            // at the end of every other TP1 state.
            if( clear_apply )
            begin
                // apply has been evaluated. save buf_idx for programming
                APPLY_CACHE <= 'd0;
                BUF_IDX <= APPLY_CACHE[1] ? 'd1 : 'd0;
            end
            else if( APPLY[0] && !apply_is_pending )
            begin
                APPLY_CACHE <= (1'b1 << 0);
            end
            else if( APPLY[1] && !apply_is_pending )
            begin
                APPLY_CACHE <= (1'b1 << 1);
            end

            LOADING <= ((state == STATE_PROG) << BUF_IDX);
        end
    end


    always @(*)
    begin
        case( state )
            STATE_IDLE:
            begin
                state_d = ENABLE ? STATE_INIT_TP1_1H : state;
            end

            STATE_INIT_TP1_1H:
            begin
                state_d = tp1_pw_done ? STATE_INIT_TP1_1L : state;
            end

            STATE_INIT_TP1_1L:
            begin
                state_d = tp1_pw_done ? STATE_INIT_TP1_2H : state;
            end

            STATE_INIT_TP1_2H:
            begin
                state_d = tp1_pw_done ? STATE_INIT_TP1_2L : state;
            end

            STATE_INIT_TP1_2L:
            begin
                state_d = tp1_pw_done ? STATE_INIT_TP1_3H : state;
            end

            STATE_INIT_TP1_3H:
            begin
                state_d = tp1_pw_done ? STATE_INIT_PROG : state;
            end

            STATE_INIT_PROG:
            begin
                state_d = PROG_DONE ? STATE_INIT_TP1_4H : state;
            end

            STATE_INIT_TP1_4H:
            begin
                state_d = tp1_pw_done ? STATE_DONE : state;
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
                state_d = tp1_period_expired ? STATE_POL : state;
            end

            STATE_POL:
            begin
                state_d = pol_done ? STATE_TP1 : state;
            end

            STATE_TP1:
            begin
                if( tp1_pw_done )
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
                state_d = tp1_period_expired ? STATE_POL_FIN : state;
            end

            STATE_POL_FIN:
            begin
                state_d = pol_done ? STATE_TP1_FIN : state;
            end

            STATE_TP1_FIN:
            begin
                if( tp1_pw_done )
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
    assign USE_RESET_CODE = (state == STATE_INIT_PROG) ||
                            (state == STATE_PROG_FIN);

    assign STATE_ONEHOT = (1'b1 << state);

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
            POL <= POL_OVR_EN ? POL_OVR_VAL : ~POL;
            phase <= ~phase;
        end
        else if( state == STATE_POL_FIN && cnt == 'd0 )
        begin
            // only update once per POL state
            // * For Bravo, POL deviates from phase (only) when finishing such
            //   that POL=0 and coefficients of all-ones causes 0V on all (odd)
            //   channels.
            // * For Delta, POL never deviates from phase such that all-zeros
            //   coefficients cause 9V on all channels when finishing.
            POL <= POL_FINISH_OVR ? 1'b0 : ~POL;
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
                   (state == STATE_INIT_TP1_4H ) ||
                   (state == STATE_TP1         ) ||
                   (state == STATE_TP1_FIN     ) ||
                   (state == STATE_DONE && TP1_DONE_HIGH);

            if (state == STATE_DONE || state == STATE_IDLE)
            begin
                ito_cnt <= 'd0;
                ITO_CLK <= 1'b0;
            end
            else if (ITO_ASYNC)
            begin
                if (ito_cnt == ITO_TC)
                begin
                    ito_cnt <= 'd0;
                    ITO_CLK <= ~ITO_CLK;
                end
                else
                begin
                    ito_cnt <= ito_cnt + 'd1;
                end
            end
            else // if (!ITO_ASYNC)
            begin
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
                    end
                end
            end
        end
    end

endmodule
