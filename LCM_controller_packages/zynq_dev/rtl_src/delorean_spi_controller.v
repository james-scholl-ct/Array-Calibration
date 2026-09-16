`include "delorean_defines.v"
`timescale 1 ns / 1 ps

module delorean_spi_controller #(
    parameter integer BASE_FRAME_N_BITS = 16,   // must be a power of 2
    parameter integer CLK_DIV_N_BITS = 4,
    parameter integer MAX_BASE_FRAMES = 16,
    parameter integer N_SLAVES = 8
)
(
    input   wire                                        S_AXI_ACLK,
    input   wire                                        S_AXI_ARESETN,

    // Host side
    input   wire    [CLK_DIV_N_BITS - 1 : 0]            CLK_DIV,
    input   wire    [$clog2(MAX_BASE_FRAMES) - 1 : 0]   N_BASE_FRAMES, // 0-15 -> 1-16
    input   wire    [$clog2(N_SLAVES) - 1 : 0]          SLAVE_IDX,
    input   wire                                        START,
    input   wire    [BASE_FRAME_N_BITS - 1 : 0]         WDATA_0,
    input   wire    [BASE_FRAME_N_BITS - 1 : 0]         WDATA_1,

    output  reg                                         DONE,
    output  reg     [BASE_FRAME_N_BITS - 1 : 0]         RDATA_0,
    output  reg     [BASE_FRAME_N_BITS - 1 : 0]         RDATA_1,
    output  wire                                        WDATA_REQ,

    // Peripheral side
    input   wire    [1 : 0]                             MISO,
    output  reg     [1 : 0]                             MOSI,
    output  reg                                         SCLK,
    output  reg     [N_SLAVES - 1 : 0]                  SS_B
);


    localparam integer STATE_N_BITS = 2;
    localparam [STATE_N_BITS - 1 : 0] STATE_IDLE   = 'd0;
    localparam [STATE_N_BITS - 1 : 0] STATE_SS_ON  = 'd1;
    localparam [STATE_N_BITS - 1 : 0] STATE_DATA   = 'd2;
    localparam [STATE_N_BITS - 1 : 0] STATE_SS_OFF = 'd3;

    reg [STATE_N_BITS - 1 : 0] state;


    // ------------------------------------------------------------------------
    // Clock divider
    // ------------------------------------------------------------------------
    // freq = 100MHz / (CLK_DIV + 1) if CLK_DIV > 0 else 50MHz
    reg  [CLK_DIV_N_BITS - 1 : 0]       cnt;
    wire [CLK_DIV_N_BITS - 1 : 0]       clk_div_int;
    wire                                sclk_posedge;
    wire                                sclk_negedge;

    // prevent illegal corner case by forcing divide by 2
    assign clk_div_int = (CLK_DIV == 'd0) ? 'd1 : CLK_DIV;

    assign sclk_posedge = (cnt == clk_div_int >> 1);

    assign sclk_negedge = (cnt == clk_div_int);

    always @(posedge S_AXI_ACLK)
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            cnt <= 'd0;
        end
        else if( sclk_negedge || state == STATE_IDLE )
        begin
            cnt <= 'd0;
        end
        else
        begin
            cnt <= cnt + 'd1;
        end
    end

    // ------------------------------------------------------------------------
    // FSM
    // ------------------------------------------------------------------------
    // frame_cnt needs to be == MAX_BASE_FRAMES so add 1 element. For 1 frame,
    // N_BASE_FRAMES is 0 and frame_cnt goes from 0 to 1, inclusive. Also,
    // for 16 frames, N_BASE_FRAMES is 15 and frame_cnt goes from 0 to 16.
    reg     [$clog2(BASE_FRAME_N_BITS) - 1 : 0]     bit_cnt;
    wire                                            bit_cnt_tc;
    reg     [$clog2(MAX_BASE_FRAMES + 1) - 1 : 0]   frame_cnt;
    wire                                            frame_cnt_tc;

    assign bit_cnt_tc = (bit_cnt == BASE_FRAME_N_BITS - 'd1);

    assign frame_cnt_tc = (frame_cnt == N_BASE_FRAMES + 'd1);


    always @(posedge S_AXI_ACLK)
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            bit_cnt <= 'd0;
            frame_cnt <= 'd0;
        end
        else if( state == STATE_IDLE )
        begin
            bit_cnt <= 'd0;
            frame_cnt <= 'd0;
        end
        else
        begin
            if( sclk_negedge )
                bit_cnt <= bit_cnt + 'd1;
            if( sclk_negedge && bit_cnt_tc )
                frame_cnt <= frame_cnt + 'd1;
        end
    end


    always @(posedge S_AXI_ACLK)
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            state <= STATE_IDLE;
        end
        else
        begin
            case(state)
                STATE_IDLE:
                begin
                    state <= START ? STATE_SS_ON : state;
                end

                STATE_SS_ON:
                begin
                    state <= sclk_negedge ? STATE_DATA : state;
                end

                STATE_DATA:
                begin
                    state <= sclk_negedge && frame_cnt_tc ? STATE_SS_OFF : state;
                end

                STATE_SS_OFF:
                begin
                    state <= sclk_negedge ? STATE_IDLE : state;
                end
            endcase
        end
    end

    // ------------------------------------------------------------------------
    // SCLK
    // ------------------------------------------------------------------------
    always @(posedge S_AXI_ACLK)
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            SCLK <= 'd0;
        end
        else if( state != STATE_DATA )
        begin
            SCLK <= 1'b0;
        end
        else if( sclk_posedge )
        begin
            SCLK <= 1'b1;
        end
        else if( sclk_negedge )
        begin
            SCLK <= 1'b0;
        end
    end

    // ------------------------------------------------------------------------
    // SS (slave select)
    // ------------------------------------------------------------------------
    always @(posedge S_AXI_ACLK)
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            SS_B <= {N_SLAVES{1'b1}};
        end
        else if( state == STATE_SS_OFF && sclk_negedge )
        begin
            SS_B <= {N_SLAVES{1'b1}};
        end
        else if( state == STATE_SS_ON && sclk_negedge )
        begin
            SS_B[SLAVE_IDX] <= 1'b0;
        end
    end

    // ------------------------------------------------------------------------
    // MOSI (master out, slave in)
    // ------------------------------------------------------------------------
    reg  [BASE_FRAME_N_BITS - 1 : 0]            tx_data_0;
    reg  [BASE_FRAME_N_BITS - 1 : 0]            tx_data_1;
    wire                                        load_tx_data;
    wire [$clog2(BASE_FRAME_N_BITS) - 1 : 0]    bit_cnt_rev;

    assign load_tx_data = (state == STATE_IDLE) && START ||
                          (state == STATE_DATA) && sclk_negedge && bit_cnt_tc;

    assign WDATA_REQ = (state == STATE_DATA) && sclk_posedge && bit_cnt_tc;

    assign bit_cnt_rev = ~bit_cnt;


    always @(posedge S_AXI_ACLK)
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            tx_data_0 <= 'd0;
            tx_data_1 <= 'd0;
        end
        else if( load_tx_data )
        begin
            tx_data_0 <= WDATA_0;
            tx_data_1 <= WDATA_1;
        end
    end


    always @(posedge S_AXI_ACLK)
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            MOSI <= 'd0;
        end
        else if( sclk_negedge )
        begin
            if( state == STATE_SS_ON || state == STATE_DATA )
            begin
                MOSI[0] <= frame_cnt_tc ? 1'b0 : tx_data_0[bit_cnt_rev];
                MOSI[1] <= frame_cnt_tc ? 1'b0 : tx_data_1[bit_cnt_rev];
            end
            else
            begin
                MOSI <= 'd0;
            end
        end
    end

    // ------------------------------------------------------------------------
    // MISO (master in, slave out)
    // ------------------------------------------------------------------------
    always @(posedge S_AXI_ACLK)
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            DONE <= 1'b0;
            RDATA_0 <= 'd0;
            RDATA_1 <= 'd0;
        end
        else
        begin
            DONE <= (state == STATE_SS_OFF && sclk_negedge);

            if(state == STATE_DATA && sclk_negedge)
            begin
                RDATA_0 <= {RDATA_0[BASE_FRAME_N_BITS - 2 : 0], MISO[0]};
                RDATA_1 <= {RDATA_1[BASE_FRAME_N_BITS - 2 : 0], MISO[1]};
            end
        end
    end

endmodule
