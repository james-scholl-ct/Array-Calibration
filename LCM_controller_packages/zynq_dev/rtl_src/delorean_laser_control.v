`include "delorean_defines.v"
`timescale 1 ns / 1 ps

module delorean_laser_control #(
    parameter integer LASER_PW_SEL_N_BITS = 8,
    parameter integer CLKS_PER_INTERVAL_N_BITS = 8,
    parameter integer PULSES_PER_FRAME_N_BITS = 1,
    parameter integer INTERVALS_PER_FRAME_N_BITS = 8
)
(
    input   wire                                        S_AXI_ACLK,
    input   wire                                        S_AXI_ARESETN,
    input   wire                                        LASER_ENABLE,
    input   wire                                        LASER_START,
    input   wire    [LASER_PW_SEL_N_BITS - 1 : 0]       LASER_PW_SEL,
    input   wire    [CLKS_PER_INTERVAL_N_BITS - 1 : 0]  CLKS_PER_INTERVAL,
    input   wire    [PULSES_PER_FRAME_N_BITS - 1 : 0]   PULSES_PER_FRAME,
    input   wire    [INTERVALS_PER_FRAME_N_BITS - 1 : 0] INTERVALS_PER_FRAME,

    output  wire                                        LASER_DR1,
    output  wire                                        LASER_DR2,
    output  wire                                        LASER_TRIGGER
);


    localparam integer CNT_N_BITS = CLKS_PER_INTERVAL_N_BITS;
    localparam integer INTERVAL_CNT_N_BITS = INTERVALS_PER_FRAME_N_BITS;

    localparam integer STATE_N_BITS = 2;
    localparam [STATE_N_BITS - 1 : 0] STATE_IDLE = 'd0;
    localparam [STATE_N_BITS - 1 : 0] STATE_FIRE = 'd1;
    localparam [STATE_N_BITS - 1 : 0] STATE_WAIT = 'd2;


    // Control logic
    reg     [STATE_N_BITS - 1 : 0]          state;
    reg     [CNT_N_BITS - 1 : 0]            cnt;
    reg     [INTERVAL_CNT_N_BITS - 1 : 0]   interval_cnt;
    wire                                    interval_done;
    wire                                    pulses_done;
    wire                                    frame_done;
    wire                                    continuous_mode;

    assign interval_done = (cnt == CLKS_PER_INTERVAL);

    assign pulses_done = (interval_cnt == PULSES_PER_FRAME);

    assign frame_done = (interval_cnt == INTERVALS_PER_FRAME);

    // only care about frame size if running continuously so use a null value
    // to disable the continuous feature.
    assign continuous_mode = (INTERVALS_PER_FRAME != 'd0);


    always @(posedge S_AXI_ACLK)
    begin
        if (S_AXI_ARESETN == 1'b0)
        begin
            cnt <= 'd0;
        end
        else if (interval_done || state == STATE_IDLE)
        begin
            cnt <= 'd0;
        end
        else
        begin
            cnt <= cnt + 'd1;
        end
    end


    always @(posedge S_AXI_ACLK)
    begin
        if (S_AXI_ARESETN == 1'b0)
        begin
            state <= STATE_IDLE;
            interval_cnt <= 'd0;
        end
        else
        begin
            case (state)
                STATE_IDLE:
                begin
                    if (continuous_mode)
                    begin
                        // use level to begin continuous mode
                        if (LASER_ENABLE)
                        begin
                            state <= STATE_FIRE;
                            interval_cnt <= 'd0;
                        end
                    end
                    else
                    begin
                        // use pulse to begin single-shot mode
                        if (LASER_START)
                        begin
                            state <= STATE_FIRE;
                            interval_cnt <= 'd0;
                        end
                    end
                end

                STATE_FIRE:
                begin
                    if (continuous_mode)
                    begin
                        if (interval_done && pulses_done && frame_done)
                        begin
                            // re-fire if LASER_ENABLE is still asserted, else stop
                            state <= LASER_ENABLE ? STATE_FIRE : STATE_IDLE;
                            interval_cnt <= 'd0;
                        end
                        else if (interval_done && pulses_done && !frame_done)
                        begin
                            state <= STATE_WAIT;
                            interval_cnt <= interval_cnt + 'd1;
                        end
                        else if (interval_done && !pulses_done)
                        begin
                            interval_cnt <= interval_cnt + 'd1;
                        end
                    end
                    else
                    begin
                        if (interval_done && pulses_done)
                        begin
                            state <= STATE_IDLE;
                            interval_cnt <= 'd0;
                        end
                        else if (interval_done && !pulses_done)
                        begin
                            interval_cnt <= interval_cnt + 'd1;
                        end
                    end
                end

                STATE_WAIT:
                begin
                    // only for continuous mode
                    if (interval_done && frame_done)
                    begin
                        // re-fire if LASER_ENABLE is still asserted, else stop
                        state <= LASER_ENABLE ? STATE_FIRE : STATE_IDLE;
                        interval_cnt <= 'd0;
                    end
                    else if (interval_done && !frame_done)
                    begin
                        state <= STATE_WAIT;
                        interval_cnt <= interval_cnt + 'd1;
                    end
                end

                default:
                begin
                    state <= STATE_IDLE;
                    interval_cnt <= 'd0;
                end
            endcase
        end
    end


    // Pulse generation circuits
    //      The ic Haus driver (TE163 and TE124) uses two LVDS pairs with phase
    //      delay to generate a pulse as follows: LASER_DR1 && !LASER_DR2. This
    //      controller stretches the fire pulse to 3 cycles (30 ns) so,
    //      theoretically, a pulse of up to 30 ns can be requested. The ic Haus
    //      driver (TE163 and TE124) internally limits all pulses to 40 ns.
    reg          pulse;
    reg  [2 : 0] pulse_dly;
    reg          pulse_or;

    always @(posedge S_AXI_ACLK)
    begin
        if (S_AXI_ARESETN == 1'b0)
        begin
            pulse       <= 1'b0;
            pulse_dly   <= 'd0;
            pulse_or    <= 1'b0;
        end
        else
        begin
            pulse       <= interval_done && (state == STATE_FIRE);
            pulse_dly   <= {pulse_dly[1:0], pulse};
            pulse_or    <= (| pulse_dly);
        end
    end

    // Each inverter stage takes about 0.350ns (prop = 0.125ns, net = 0.225ns).
    // Tap every four to maintain polarity and produce about 1.4 ns per tap.
    localparam integer N_INV_PER_TAP = 4;
    localparam integer N_INV_PW = N_INV_PER_TAP * (LASER_PW_SEL_N_BITS - 1);

    (* dont_touch = "true" *) wire [N_INV_PW : 0]                 pw_inv_line;
                              wire [LASER_PW_SEL_N_BITS - 1 : 0]  pw_tap;
    (* dont_touch = "true" *) wire [LASER_PW_SEL_N_BITS - 1 : 0]  pw_gate;

    assign pw_inv_line = {~pw_inv_line[N_INV_PW - 1 : 0], pulse_or};

    genvar i_gv;
    generate
        for (i_gv = 0; i_gv < LASER_PW_SEL_N_BITS; i_gv = i_gv + 1)
        begin : gen_pw_tap
            assign pw_tap[i_gv] = pw_inv_line[i_gv * N_INV_PER_TAP];
        end
    endgenerate

    assign pw_gate = LASER_PW_SEL & pw_tap;

    assign LASER_DR1 = pulse_or;
    assign LASER_DR2 = |pw_gate;
    assign LASER_TRIGGER = pulse_or;

endmodule
