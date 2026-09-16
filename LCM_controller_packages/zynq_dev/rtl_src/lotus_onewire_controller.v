`include "lotus_defines.v"
`timescale 1 ns / 1 ps

module lotus_onewire_controller #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer N_SLAVES = 3
)
(
    // Host side
    input   wire                                        S_AXI_ACLK,
    input   wire                                        S_AXI_ARESETN,
    output  reg     [N_SLAVES * C_S_AXI_DATA_WIDTH - 1 : 0] RDATA,

    // Slave side
    input   wire    [N_SLAVES - 1 : 0]                  OW_RX_DATA,
    output  reg                                         OW_PULLUP_EN_B,
    output  reg     [N_SLAVES - 1 : 0]                  OW_TX_DATA
);


    localparam integer CNT_N_BITS = 7;
    localparam [CNT_N_BITS - 1 : 0] CNT_EQ_1US = 'd99; // 1us

    localparam integer US_CNT_N_BITS = 20; // 1048.576 ms > 750ms
    localparam [US_CNT_N_BITS - 1 : 0] T_RESET_US = 'd500;
    localparam [US_CNT_N_BITS - 1 : 0] T_PRESENCE_SMPL_US = 'd70;
    localparam [US_CNT_N_BITS - 1 : 0] T_SLOT_US = 'd100;
    localparam [US_CNT_N_BITS - 1 : 0] T_WR0_PD_US = 'd80;
    localparam [US_CNT_N_BITS - 1 : 0] T_WR1_PD_US = 'd5;
    localparam [US_CNT_N_BITS - 1 : 0] T_RD_PD_US = 'd2;
    localparam [US_CNT_N_BITS - 1 : 0] T_RD_SMPL_US = 'd14;
    localparam [US_CNT_N_BITS - 1 : 0] T_CONVERT_US = 'd750000;

    localparam integer STATE_N_BITS = 4;
    localparam [STATE_N_BITS - 1 : 0] STATE_IDLE            = 'd0;
    localparam [STATE_N_BITS - 1 : 0] STATE_RESET_0         = 'd1;
    localparam [STATE_N_BITS - 1 : 0] STATE_PRESENCE_0      = 'd2;
    localparam [STATE_N_BITS - 1 : 0] STATE_TX_SKIPROM_0    = 'd3;
    localparam [STATE_N_BITS - 1 : 0] STATE_TX_CONVERT      = 'd4;
    localparam [STATE_N_BITS - 1 : 0] STATE_WAIT_CONVERT    = 'd5;
    localparam [STATE_N_BITS - 1 : 0] STATE_RESET_1         = 'd6;
    localparam [STATE_N_BITS - 1 : 0] STATE_PRESENCE_1      = 'd7;
    localparam [STATE_N_BITS - 1 : 0] STATE_TX_SKIPROM_1    = 'd8;
    localparam [STATE_N_BITS - 1 : 0] STATE_TX_READSCRATCH  = 'd9;
    localparam [STATE_N_BITS - 1 : 0] STATE_RX_READSCRATCH  = 'd10;
    localparam [STATE_N_BITS - 1 : 0] STATE_DONE            = 'd11;

    localparam integer BIT_CNT_N_BITS = 7;

    localparam integer RDATA_N_BITS = N_SLAVES * C_S_AXI_DATA_WIDTH;

    wire enable = 1'b1;

    //
    reg     [STATE_N_BITS - 1 : 0]          state;
    reg     [CNT_N_BITS - 1 : 0]            cnt;
    reg     [US_CNT_N_BITS - 1 : 0]         us_cnt;
    reg     [BIT_CNT_N_BITS - 1 : 0]        bit_cnt;
    wire                                    us_done;

    assign us_done = (cnt == CNT_EQ_1US);

    always @( posedge S_AXI_ACLK )
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            cnt <= 'd0;
        end
        else if( !enable )
        begin
            cnt <= 'd0;
        end
        else
        begin
            cnt <= us_done ? 'd0 : cnt + 'd1;
        end
    end


    always @( posedge S_AXI_ACLK )
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            state <= STATE_IDLE;
            us_cnt <= 'd0;
            bit_cnt <= 'd0;
        end
        else if( !enable )
        begin
            state <= STATE_IDLE;
            us_cnt <= 'd0;
            bit_cnt <= 'd0;
        end
        else if( us_done )
        begin
            case( state )
                STATE_IDLE:
                begin
                    if( us_cnt == T_RESET_US - 'd1 )
                    begin
                        state <= STATE_RESET_0;
                        us_cnt <= 'd0;
                        bit_cnt <= 'd0;
                    end
                    else
                    begin
                        state <= state;
                        us_cnt <= us_cnt + 'd1;
                        bit_cnt <= 'd0;
                    end
                end

                STATE_RESET_0:
                begin
                    if( us_cnt == T_RESET_US - 'd1 )
                    begin
                        state <= STATE_PRESENCE_0;
                        us_cnt <= 'd0;
                        bit_cnt <= 'd0;
                    end
                    else
                    begin
                        state <= state;
                        us_cnt <= us_cnt + 'd1;
                        bit_cnt <= 'd0;
                    end
                end

                STATE_PRESENCE_0:
                begin
                    if( us_cnt == T_RESET_US - 'd1 )
                    begin
                        state <= STATE_TX_SKIPROM_0;
                        us_cnt <= 'd0;
                        bit_cnt <= 'd0;
                    end
                    else
                    begin
                        state <= state;
                        us_cnt <= us_cnt + 'd1;
                        bit_cnt <= 'd0;
                    end
                end

                STATE_TX_SKIPROM_0:
                begin
                    if( us_cnt == T_SLOT_US - 'd1 )
                    begin
                        us_cnt <= 'd0;
                        if( bit_cnt == 'd7 )
                        begin
                            state <= STATE_TX_CONVERT;
                            bit_cnt <= 'd0;
                        end
                        else
                        begin
                            state <= state;
                            bit_cnt <= bit_cnt + 'd1;
                        end
                    end
                    else
                    begin
                        state <= state;
                        us_cnt <= us_cnt + 'd1;
                        bit_cnt <= bit_cnt;
                    end
                end

                STATE_TX_CONVERT:
                begin
                    if( us_cnt == T_SLOT_US - 'd1 )
                    begin
                        us_cnt <= 'd0;
                        if( bit_cnt == 'd7 )
                        begin
                            state <= STATE_WAIT_CONVERT;
                            bit_cnt <= 'd0;
                        end
                        else
                        begin
                            state <= state;
                            bit_cnt <= bit_cnt + 'd1;
                        end
                    end
                    else
                    begin
                        state <= state;
                        us_cnt <= us_cnt + 'd1;
                        bit_cnt <= bit_cnt;
                    end
                end

                STATE_WAIT_CONVERT:
                begin
                    if( us_cnt == T_CONVERT_US - 'd1 )
                    begin
                        state <= STATE_RESET_1;
                        us_cnt <= 'd0;
                        bit_cnt <= 'd0;
                    end
                    else
                    begin
                        state <= state;
                        us_cnt <= us_cnt + 'd1;
                        bit_cnt <= 'd0;
                    end
                end

                STATE_RESET_1:
                begin
                    if( us_cnt == T_RESET_US - 'd1 )
                    begin
                        state <= STATE_PRESENCE_1;
                        us_cnt <= 'd0;
                        bit_cnt <= 'd0;
                    end
                    else
                    begin
                        state <= state;
                        us_cnt <= us_cnt + 'd1;
                        bit_cnt <= 'd0;
                    end
                end

                STATE_PRESENCE_1:
                begin
                    if( us_cnt == T_RESET_US - 'd1 )
                    begin
                        state <= STATE_TX_SKIPROM_1;
                        us_cnt <= 'd0;
                        bit_cnt <= 'd0;
                    end
                    else
                    begin
                        state <= state;
                        us_cnt <= us_cnt + 'd1;
                        bit_cnt <= 'd0;
                    end
                end

                STATE_TX_SKIPROM_1:
                begin
                    if( us_cnt == T_SLOT_US - 'd1 )
                    begin
                        us_cnt <= 'd0;
                        if( bit_cnt == 'd7 )
                        begin
                            state <= STATE_TX_READSCRATCH;
                            bit_cnt <= 'd0;
                        end
                        else
                        begin
                            state <= state;
                            bit_cnt <= bit_cnt + 'd1;
                        end
                    end
                    else
                    begin
                        state <= state;
                        us_cnt <= us_cnt + 'd1;
                        bit_cnt <= bit_cnt;
                    end
                end

                STATE_TX_READSCRATCH:
                begin
                    if( us_cnt == T_SLOT_US - 'd1 )
                    begin
                        us_cnt <= 'd0;
                        if( bit_cnt == 'd7 )
                        begin
                            state <= STATE_RX_READSCRATCH;
                            bit_cnt <= 'd0;
                        end
                        else
                        begin
                            state <= state;
                            bit_cnt <= bit_cnt + 'd1;
                        end
                    end
                    else
                    begin
                        state <= state;
                        us_cnt <= us_cnt + 'd1;
                        bit_cnt <= bit_cnt;
                    end
                end

                STATE_RX_READSCRATCH:
                begin
                    if( us_cnt == T_SLOT_US - 'd1 )
                    begin
                        us_cnt <= 'd0;
                        if( bit_cnt == 'd71 )
                        begin
                            // 9 bytes
                            state <= STATE_DONE;
                            bit_cnt <= 'd0;
                        end
                        else
                        begin
                            state <= state;
                            bit_cnt <= bit_cnt + 'd1;
                        end
                    end
                    else
                    begin
                        state <= state;
                        us_cnt <= us_cnt + 'd1;
                        bit_cnt <= bit_cnt;
                    end
                end

                STATE_DONE:
                begin
                    state <= STATE_IDLE;
                    us_cnt <= 'd0;
                    bit_cnt <= 'd0;
                end

                default:
                begin
                    state <= STATE_IDLE;
                    us_cnt <= 'd0;
                    bit_cnt <= 'd0;
                end
            endcase
        end
    end


    // ------------------------------------------------------------------------
    // TX Datapath
    // ------------------------------------------------------------------------
    wire    [7 : 0] wbuff;
    wire            wdata;
    wire            write_pd;
    wire            read_pd;
    wire            pull_down;
    wire            strong_pull_up;


    assign wbuff = (state == STATE_TX_SKIPROM_0)     ? 8'hcc :
                   (state == STATE_TX_SKIPROM_1)     ? 8'hcc :
                   (state == STATE_TX_CONVERT)       ? 8'h44 :
                   (state == STATE_TX_READSCRATCH)   ? 8'hBE : 8'hFF;

    // LSB first
    assign wdata = wbuff[bit_cnt[2:0]];

    assign write_pd = wdata ? (us_cnt < T_WR1_PD_US) :
                              (us_cnt < T_WR0_PD_US) ;

    assign read_pd = (us_cnt < T_RD_PD_US);

    assign pull_down = (state == STATE_RESET_0) ||
                       (state == STATE_RESET_1) ||
                       (state == STATE_TX_SKIPROM_0   && write_pd) ||
                       (state == STATE_TX_SKIPROM_1   && write_pd) ||
                       (state == STATE_TX_CONVERT     && write_pd) ||
                       (state == STATE_TX_READSCRATCH && write_pd) ||
                       (state == STATE_RX_READSCRATCH && read_pd);

    assign strong_pull_up = (state == STATE_WAIT_CONVERT);

    always @( posedge S_AXI_ACLK )
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            OW_TX_DATA <= {N_SLAVES{1'b1}};
            OW_PULLUP_EN_B <= 1'b1;
        end
        else
        begin
            OW_TX_DATA <= {N_SLAVES{!pull_down}};
            OW_PULLUP_EN_B <= !strong_pull_up;
        end
    end

    // ------------------------------------------------------------------------
    // RX Datapath
    // ------------------------------------------------------------------------
    genvar              i_gv;
    wire                sample_presence;
    wire                sample_scratchpad;
    reg                 rx_sync1 [N_SLAVES - 1 : 0];
    reg                 rx_sync2 [N_SLAVES - 1 : 0];
    reg     [1 : 0]     shiftreg [N_SLAVES - 1 : 0];
    wire    [1 : 0]     maj_vote [N_SLAVES - 1 : 0];
    reg                 din      [N_SLAVES - 1 : 0];
    reg                 presence [N_SLAVES - 1 : 0];
    reg     [71 : 0]    rbuff    [N_SLAVES - 1 : 0];
    reg     [7 : 0]     crc      [N_SLAVES - 1 : 0];
    wire                crc_fb   [N_SLAVES - 1 : 0];
    wire    [7 : 0]     crc_xor  [N_SLAVES - 1 : 0];
    //wire                rx_error [N_SLAVES - 1 : 0];
    wire    [N_SLAVES - 1 : 0] rx_error;

    wire    [C_S_AXI_DATA_WIDTH - 1 : 0] rdata_unpacked [N_SLAVES - 1 : 0];
    wire    [RDATA_N_BITS - 1 : 0]       rdata_packed;

    assign sample_presence = us_done && (us_cnt == T_PRESENCE_SMPL_US - 'd1) &&
                             (state == STATE_PRESENCE_0 ||
                              state == STATE_PRESENCE_1);

    assign sample_scratchpad = us_done && (us_cnt == T_RD_SMPL_US - 'd1) &&
                               (state == STATE_RX_READSCRATCH);

    generate
        for(i_gv = 0; i_gv < N_SLAVES; i_gv = i_gv + 1)
        begin : gen_rx_datapath
            // CDC
            always @( posedge S_AXI_ACLK )
            begin
                rx_sync1[i_gv] <= OW_RX_DATA[i_gv];
                rx_sync2[i_gv] <= rx_sync1[i_gv];
            end

            // Detection
            always @( posedge S_AXI_ACLK )
            begin
                if( S_AXI_ARESETN == 1'b0 )
                begin
                    shiftreg[i_gv] <= 'd0;
                    din[i_gv] <= 1'b0;
                end
                else
                begin
                    shiftreg[i_gv] <= {shiftreg[i_gv][0], rx_sync2[i_gv]};
                    din[i_gv] <= (maj_vote[i_gv] > 'd1);
                end
            end

            assign maj_vote[i_gv] = shiftreg[i_gv][1] +
                                    shiftreg[i_gv][0] +
                                    rx_sync2[i_gv];

            // Presence
            //  Accumulate all presence samples until DONE and reset in IDLE
            always @( posedge S_AXI_ACLK )
            begin
                if( S_AXI_ARESETN == 1'b0 )
                begin
                    presence[i_gv] <= 1'b1;
                end
                else if(state == STATE_IDLE)
                begin
                    presence[i_gv] <= 1'b1;
                end
                else if( sample_presence )
                begin
                    presence[i_gv] <= presence[i_gv] & ~din[i_gv];
                end
            end

            // RDATA and CRC (x**8 + x**5 + x**4 + 1)
            assign crc_fb[i_gv] = crc[i_gv][7] ^ din[i_gv];

            assign crc_xor[i_gv] = {2'b00, {2{crc_fb[i_gv]}}, 3'b000, crc_fb[i_gv]};

            always @( posedge S_AXI_ACLK )
            begin
                if( S_AXI_ARESETN == 1'b0 )
                begin
                    rbuff[i_gv] <= 72'd0;
                    crc[i_gv] <= 'd0;
                end
                else if(state == STATE_IDLE)
                begin
                    rbuff[i_gv] <= 72'd0;
                    crc[i_gv] <= 'd0;
                end
                else if( sample_scratchpad )
                begin
                    // data sent LSB first so reverse it
                    rbuff[i_gv] <= {din[i_gv], rbuff[i_gv][71 : 1]};
                    crc[i_gv] <= (crc[i_gv] << 1) ^ crc_xor[i_gv];
                end
            end

            assign rx_error[i_gv] = !presence[i_gv] || (crc[i_gv] != 'd0);

            assign rdata_unpacked[i_gv] = {19'd0, !rx_error[i_gv], rbuff[i_gv][11 : 0]};

            assign rdata_packed[i_gv * C_S_AXI_DATA_WIDTH +: C_S_AXI_DATA_WIDTH] =
                   rdata_unpacked[i_gv];
        end
    endgenerate

    always @( posedge S_AXI_ACLK )
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            RDATA <= {RDATA_N_BITS{1'b0}};
        end
        else if( state == STATE_DONE && us_done )
        begin
            RDATA <= rdata_packed;
        end
    end

endmodule
