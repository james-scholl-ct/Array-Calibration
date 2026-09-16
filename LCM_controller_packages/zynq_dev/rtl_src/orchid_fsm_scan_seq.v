`timescale 1 ns / 1 ps

module orchid_fsm_scan_seq #(
    parameter CNT_N_BITS = 'd20,        // up to 8.7ms @ 120MHz
    parameter N_CYCLES_TP1 = 'd60,      // 500ns @ 120MHz
    parameter N_CYCLES_POL = 'd2        // > 6ns @ 120MHz
)
(
    input                               CLK,
    input       [ CNT_N_BITS - 1 : 0 ]  DWELL_N_CYCLES_MODE00,
    input       [ CNT_N_BITS - 1 : 0 ]  DWELL_N_CYCLES_NORMAL,
    input       [ 1 : 0 ]               MODE,
    input                               PROG_DONE,
    input                               RESET,
    input                               START,

    output                              FORCE_ZEROS,
    output reg                          POL,
    output                              PROG_START,
    output reg  [ 1 : 0 ]               TABLE_SEL,
    output reg                          TP1
);


    localparam [ 4 : 0 ] STATE_IDLE        = 'd0;
    localparam [ 4 : 0 ] STATE_INIT_TP1_1A = 'd1;
    localparam [ 4 : 0 ] STATE_INIT_TP1_1B = 'd2;
    localparam [ 4 : 0 ] STATE_INIT_TP1_2A = 'd3;
    localparam [ 4 : 0 ] STATE_INIT_TP1_2B = 'd4;
    localparam [ 4 : 0 ] STATE_INIT_TP1_3A = 'd5;
    localparam [ 4 : 0 ] STATE_INIT_TP1_3B = 'd6;
    localparam [ 4 : 0 ] STATE_INIT_PROG   = 'd7;
    localparam [ 4 : 0 ] STATE_INIT_POL    = 'd8;
    localparam [ 4 : 0 ] STATE_INIT_TP1    = 'd9;
    localparam [ 4 : 0 ] STATE_WAIT        = 'd10;
    localparam [ 4 : 0 ] STATE_REFRESH_POL = 'd11;
    localparam [ 4 : 0 ] STATE_REFRESH_TP1 = 'd12;
    localparam [ 4 : 0 ] STATE_REPROG      = 'd13;
    localparam [ 4 : 0 ] STATE_REPROG_POL  = 'd14;
    localparam [ 4 : 0 ] STATE_REPROG_TP1  = 'd15;
    localparam [ 4 : 0 ] STATE_ZEROS_PROG  = 'd16;
    localparam [ 4 : 0 ] STATE_ZEROS_POL   = 'd17;
    localparam [ 4 : 0 ] STATE_ZEROS_TP1   = 'd18;
    localparam [ 4 : 0 ] STATE_FLOAT       = 'd19;
    localparam [ 4 : 0 ] STATE_DONE        = 'd20;


    reg  [ CNT_N_BITS - 1 : 0 ] cnt;
    wire                        reset_cnt;
    wire                        pol_cnt_done;
    wire                        tp1_cnt_done;
    wire                        dwell_cnt_done;
    reg                         dwell_expired;
    reg  [ 4 : 0 ]              state;
    reg  [ 4 : 0 ]              state_d;
    reg  [ 1 : 0 ]              mode_q;
    wire                        wait_for_dwell;

    // Don't reset the counter if we are reprograming during dwell
    assign reset_cnt = (state == STATE_IDLE) ||
                       (state == STATE_DONE) ||
                       (state != state_d) && (state_d != STATE_REPROG);

    assign pol_cnt_done = (cnt == N_CYCLES_POL - 1);

    assign tp1_cnt_done = (cnt == N_CYCLES_TP1 - 1);

    assign dwell_cnt_done = (mode_q == 2'b00) ? (cnt == DWELL_N_CYCLES_MODE00 - 1) :
                                                (cnt == DWELL_N_CYCLES_NORMAL - 1) ;

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
                dwell_expired <= dwell_expired || dwell_cnt_done;
            end
        end
    end


    assign wait_for_dwell = (mode_q == 2'b00) ? 1'b1 :
                            (mode_q == 2'b01) ? 1'b0 :
                            (mode_q == 2'b10) ? ~POL :
                                                1'b0 ;


    always @( posedge CLK )
    begin
        if( RESET )
        begin
            state <= STATE_IDLE;
            mode_q <= 2'b00;
        end
        else
        begin
            state <= state_d;
            mode_q <= START ? MODE : mode_q;
        end
    end


    always @(*)
    begin
        case( state )
            STATE_IDLE:
            begin
                state_d = START ? STATE_INIT_TP1_1A : state;
            end

            STATE_INIT_TP1_1A:
            begin
                state_d = tp1_cnt_done ? STATE_INIT_TP1_1B : state;
            end

            STATE_INIT_TP1_1B:
            begin
                state_d = tp1_cnt_done ? STATE_INIT_TP1_2A : state;
            end

            STATE_INIT_TP1_2A:
            begin
                state_d = tp1_cnt_done ? STATE_INIT_TP1_2B : state;
            end

            STATE_INIT_TP1_2B:
            begin
                state_d = tp1_cnt_done ? STATE_INIT_TP1_3A : state;
            end

            STATE_INIT_TP1_3A:
            begin
                state_d = tp1_cnt_done ? STATE_INIT_TP1_3B : state;
            end

            STATE_INIT_TP1_3B:
            begin
                state_d = tp1_cnt_done ? STATE_INIT_PROG : state;
            end

            STATE_INIT_PROG:
            begin
                state_d = PROG_DONE ? STATE_INIT_POL : state;
            end

            STATE_INIT_POL:
            begin
                state_d = pol_cnt_done ? STATE_INIT_TP1 : state;
            end

            STATE_INIT_TP1:
            begin
                state_d = tp1_cnt_done ? STATE_WAIT : state;
            end

            STATE_WAIT:
            begin
                if( !START )
                begin
                    state_d = STATE_ZEROS_PROG;
                end
                else if( !wait_for_dwell )
                begin
                    state_d = STATE_REPROG;
                end
                else if( dwell_expired )
                begin
                    state_d = STATE_REFRESH_POL;
                end
                else
                begin
                    state_d = state;
                end
            end

            // -------
            // REFRESH
            // -------
            STATE_REFRESH_POL:
            begin
                state_d = pol_cnt_done ? STATE_REFRESH_TP1 : state;
            end

            STATE_REFRESH_TP1:
            begin
                state_d = tp1_cnt_done ? STATE_WAIT : state;
            end

            // ------
            // REPROG
            // ------
            STATE_REPROG:
            begin
                state_d = PROG_DONE && dwell_expired ? STATE_REPROG_POL : state;
            end

            STATE_REPROG_POL:
            begin
                state_d = pol_cnt_done ? STATE_REPROG_TP1 : state;
            end

            STATE_REPROG_TP1:
            begin
                state_d = tp1_cnt_done ? STATE_WAIT : state;
            end

            // -----
            // ZEROS
            // -----
            STATE_ZEROS_PROG:
            begin
                state_d = PROG_DONE ? STATE_ZEROS_POL : state;
            end

            STATE_ZEROS_POL:
            begin
                state_d = pol_cnt_done ? STATE_ZEROS_TP1 : state;
            end

            STATE_ZEROS_TP1:
            begin
                state_d = tp1_cnt_done ? STATE_FLOAT : state;
            end

            STATE_FLOAT:
            begin
                state_d = tp1_cnt_done ? STATE_DONE : state;
            end

            STATE_DONE:
            begin
                state_d = START ? STATE_INIT_PROG : state;
            end

            default:
            begin
                state_d = STATE_IDLE;
            end
        endcase
    end


    // Outputs
    assign FORCE_ZEROS = (state == STATE_ZEROS_PROG);


    always @( posedge CLK )
    begin
        if( RESET )
        begin
            POL <= 1'b0;
        end
        else
        begin
            // only update once per POL state
            if (cnt == 'd0)
            begin
                if( state == STATE_INIT_POL )
                begin
                    POL <= 1'b0;
                end
                else if( state == STATE_REFRESH_POL )
                begin
                    POL <= ~POL;
                end
                else if( state == STATE_REPROG_POL )
                begin
                    POL <= (mode_q == 2'b10) ? ~POL : 1'b0;
                end
                else if( state == STATE_ZEROS_POL )
                begin
                    POL <= 1'b0;
                end
            end
        end
    end


    assign PROG_START = (state == STATE_INIT_PROG)  ||
                        (state == STATE_REPROG)     ||
                        (state == STATE_ZEROS_PROG) ;


    always @( posedge CLK )
    begin
        if( RESET )
        begin
            TABLE_SEL <= 2'b00;
        end
        else
        begin
            if( state == STATE_WAIT )
            begin
                if( !START )
                begin
                    TABLE_SEL <= 2'b00;
                end
                else if( !wait_for_dwell )
                begin
                    // increment before reprogramming. only mode 3 uses four tables.
                    TABLE_SEL <= {(mode_q==2'b11), 1'b1} & (TABLE_SEL + 'd1);
                end
            end
        end
    end


    always @( posedge CLK )
    begin
        if( RESET )
        begin
            TP1 <= 1'b0;
        end
        else
        begin
            // hold TP1 high during DONE to make driver outputs float
            TP1 <= (state == STATE_INIT_TP1_1A ) ||
                   (state == STATE_INIT_TP1_2A ) ||
                   (state == STATE_INIT_TP1_3A ) ||
                   (state == STATE_INIT_TP1    ) ||
                   (state == STATE_REFRESH_TP1 ) ||
                   (state == STATE_REPROG_TP1  ) ||
                   (state == STATE_ZEROS_TP1   ) ||
                   (state == STATE_DONE        ) ;
        end
    end

endmodule
