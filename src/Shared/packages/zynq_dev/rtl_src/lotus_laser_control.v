`include "lotus_defines.v"
`timescale 1 ns / 1 ps

module lotus_laser_control #(
    parameter integer CONFIG_N_BITS = 32,
    parameter integer CONTROL_N_BITS = 32
)
(
    input   wire                                        S_AXI_ACLK,
    input   wire                                        S_AXI_ARESETN,
    input   wire    [CONFIG_N_BITS - 1 : 0]             CONFIG,
    input   wire    [CONTROL_N_BITS - 1 : 0]            CONTROL,
    input   wire                                        CONTROL_VALID,

    output  wire                                        LASER_DR1,
    output  wire                                        LASER_DR2,
    output  wire                                        LASER_TRIGGER
);


    localparam integer CNT_N_BITS = 8;
    localparam integer INTERVAL_CNT_N_BITS = 16;

    localparam integer STATE_N_BITS = 2;
    localparam [STATE_N_BITS - 1 : 0] STATE_IDLE = 'd0;
    localparam [STATE_N_BITS - 1 : 0] STATE_FIRE = 'd1;
    localparam [STATE_N_BITS - 1 : 0] STATE_WAIT = 'd2;

    localparam integer PW_SEL_N_BITS = 16;

    // Decode control and config memory
    wire                                    enable;
    wire                                    start;
    wire    [31 : 0]                        config0;
    wire    [31 : 0]                        config1;
    wire    [PW_SEL_N_BITS - 1 : 0]         pw_sel;
    wire    [7 : 0]                         trig_sel;
    wire    [CNT_N_BITS - 1 : 0]            clks_per_interval;
    wire    [7 : 0]                         pulses_per_frame;
    wire    [INTERVAL_CNT_N_BITS - 1 : 0]   intervals_per_frame;

    assign enable = CONTROL[0];

    assign start = enable && CONTROL_VALID;

    assign config0 = CONFIG[31 : 0];
    assign config1 = CONFIG[63 : 32];

    assign pw_sel = config0[PW_SEL_N_BITS - 1 : 0];

    assign trig_sel = config0[PW_SEL_N_BITS +: 8];

    assign clks_per_interval = config1[7 : 0];

    assign pulses_per_frame = config1[15 : 8];

    assign intervals_per_frame = config1[31 : 16];


    //
    reg     [STATE_N_BITS - 1 : 0]          state;
    reg     [CNT_N_BITS - 1 : 0]            cnt;
    reg     [INTERVAL_CNT_N_BITS - 1 : 0]   interval_cnt;
    wire                                    interval_done;
    wire                                    pulses_done;
    wire                                    frame_done;
    wire                                    continuous_mode;

    assign interval_done = (cnt == clks_per_interval);

    assign pulses_done = (interval_cnt == pulses_per_frame);

    assign frame_done = (interval_cnt == intervals_per_frame);

    // only care about frame size if running continuously so use a null value
    // to disable the continuous feature.
    assign continuous_mode = (intervals_per_frame != 'd0);


    always @( posedge S_AXI_ACLK )
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            cnt <= 'd0;
        end
        else if( interval_done || state == STATE_IDLE )
        begin
            cnt <= 'd0;
        end
        else
        begin
            cnt <= cnt + 'd1;
        end
    end


    always @( posedge S_AXI_ACLK )
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            state <= STATE_IDLE;
            interval_cnt <= 'd0;
        end
        else
        begin
            case( state )
                STATE_IDLE:
                begin
                    if( continuous_mode )
                    begin
                        // use level to begin continuous mode
                        if( enable )
                        begin
                            state <= STATE_FIRE;
                            interval_cnt <= 'd0;
                        end
                    end
                    else
                    begin
                        // use pulse to begin single-shot mode
                        if( start )
                        begin
                            state <= STATE_FIRE;
                            interval_cnt <= 'd0;
                        end
                    end
                end

                STATE_FIRE:
                begin
                    if( continuous_mode )
                    begin
                        if( interval_done && pulses_done && frame_done )
                        begin
                            // re-fire if enable is still asserted, else stop
                            state <= enable ? STATE_FIRE : STATE_IDLE;
                            interval_cnt <= 'd0;
                        end
                        else if( interval_done && pulses_done && !frame_done )
                        begin
                            state <= STATE_WAIT;
                            interval_cnt <= interval_cnt + 'd1;
                        end
                        else if( interval_done && !pulses_done )
                        begin
                            interval_cnt <= interval_cnt + 'd1;
                        end
                    end
                    else
                    begin
                        if( interval_done && pulses_done )
                        begin
                            state <= STATE_IDLE;
                            interval_cnt <= 'd0;
                        end
                        else if( interval_done && !pulses_done )
                        begin
                            interval_cnt <= interval_cnt + 'd1;
                        end
                    end
                end

                STATE_WAIT:
                begin
                    // only for continuous mode
                    if( interval_done && frame_done )
                    begin
                        // re-fire if enable is still asserted, else stop
                        state <= enable ? STATE_FIRE : STATE_IDLE;
                        interval_cnt <= 'd0;
                    end
                    else if( interval_done && !frame_done )
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


    reg pulse;
    reg [2 : 0] pulse_dly;
    reg pulse_or;
    always @( posedge S_AXI_ACLK )
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            pulse       <= 1'b0;
            pulse_dly   <= 'd0;
            pulse_or    <= 1'b0;
        end
        else
        begin
            pulse           <= interval_done && (state == STATE_FIRE);
            pulse_dly[0]    <= pulse;
            pulse_dly[1]    <= pulse_dly[0];
            pulse_dly[2]    <= pulse_dly[1];
            pulse_or        <= (| pulse_dly);
        end
    end


    // Each inverter stage takes about 0.350 ns (prop = 0.125ns, net = 0.225ns)
    // Use odd number of inverters.
    localparam integer N_INVERTERS = 61;

    (* dont_touch = "true" *) wire [N_INVERTERS : 0]        pw_inv_line;
                              wire [PW_SEL_N_BITS - 1 : 0]  pw_tap;
                              wire [PW_SEL_N_BITS - 1 : 0]  pw_start;
    (* dont_touch = "true" *) wire [PW_SEL_N_BITS - 1 : 0]  pw_gate;

    assign pw_inv_line = {~pw_inv_line[N_INVERTERS - 1 : 0], pulse_or};

    assign pw_tap = {pw_inv_line[61],
                     pw_inv_line[57],
                     pw_inv_line[53],
                     pw_inv_line[49],
                     pw_inv_line[45],
                     pw_inv_line[41],
                     pw_inv_line[37],
                     pw_inv_line[33],
                     pw_inv_line[29],
                     pw_inv_line[25],
                     pw_inv_line[21],
                     pw_inv_line[17],
                     pw_inv_line[13],
                     pw_inv_line[9],
                     pw_inv_line[5],
                     pw_inv_line[1]};

    assign pw_start = {PW_SEL_N_BITS{pulse_or}};

    assign pw_gate = pw_sel & pw_start & pw_tap;

    assign LASER_DR1 = |pw_gate;
    assign LASER_DR2 = |pw_gate;


    (* dont_touch = "true" *) wire [28 : 0]     trig_inv_line;
                              wire [7 : 0]      trig_tap;
    (* dont_touch = "true" *) wire [7 : 0]      trig_gate;

    assign trig_inv_line = {~trig_inv_line[28 - 1 : 0], pulse_dly[1]};

    assign trig_tap = {trig_inv_line[28],
                       trig_inv_line[24],
                       trig_inv_line[20],
                       trig_inv_line[16],
                       trig_inv_line[12],
                       trig_inv_line[8],
                       trig_inv_line[4],
                       trig_inv_line[0]};

    assign trig_gate = trig_sel & trig_tap;

    assign LASER_TRIGGER = |trig_gate;

endmodule
