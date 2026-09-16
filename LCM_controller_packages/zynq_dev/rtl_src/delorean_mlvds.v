`include "delorean_defines.v"
`timescale 1 ns / 1 ps

module delorean_mlvds #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer N_BUFFERS = 2,
    parameter integer RST_PW_N_BITS = 4,
    parameter integer STEP_N_BITS = 8,
    parameter integer TX_WAIT_N_BITS = 4
)
(
    // BRAM interface
    input   wire    [C_S_AXI_DATA_WIDTH - 1 : 0]        BRAM_DO,
    output  reg     [9 : 0]                             BRAM_ADDR,
    output  wire                                        BRAM_CLK,
    output  wire    [C_S_AXI_DATA_WIDTH - 1 : 0]        BRAM_DI,
    output  wire                                        BRAM_EN,
    output  wire    [C_S_AXI_DATA_WIDTH / 8 - 1 : 0]    BRAM_WE,

    // Non-BRAM interface
    input   wire    [7 : 0]                             AUX_CODE_EVEN,
    input   wire    [7 : 0]                             AUX_CODE_ODD,
    input   wire    [$clog2(N_BUFFERS) - 1 : 0]         BUF_IDX,
    input   wire                                        CLK,
    input   wire    [STEP_N_BITS - 1 : 0]               N_STEPS,
    input   wire                                        RESET,
    input   wire    [7 : 0]                             RESET_CODE,
    input   wire    [RST_PW_N_BITS - 1 : 0]             RST_PW,
    input   wire                                        START,
    input   wire    [TX_WAIT_N_BITS - 1 : 0]            TX_WAIT,
    input   wire                                        USE_RESET_CODE,

    output  wire                                        DONE,
    output  reg     [5 : 0]                             RX_LVDS_DATA_D1,
    output  reg     [5 : 0]                             RX_LVDS_DATA_D2,
    output  reg     [5 : 0]                             TX_LVDS_DATA_D1,
    output  reg     [5 : 0]                             TX_LVDS_DATA_D2
);


    localparam integer CNT_N_BITS = 'd4;
    localparam [CNT_N_BITS - 1 : 0] LAST_STEP_CYCLE = 'd3;

    localparam STATE_N_BITS = 'd3;
    localparam [STATE_N_BITS - 1 : 0] STATE_IDLE     = 'd0;
    localparam [STATE_N_BITS - 1 : 0] STATE_RST      = 'd1;
    localparam [STATE_N_BITS - 1 : 0] STATE_RST_WAIT = 'd2;
    localparam [STATE_N_BITS - 1 : 0] STATE_TX       = 'd3;
    localparam [STATE_N_BITS - 1 : 0] STATE_TX_WAIT  = 'd4;
    localparam [STATE_N_BITS - 1 : 0] STATE_DONE     = 'd5;

    localparam integer FRAME_SIZE = 48;

    // ------------------------------------------------------------------------
    // Control
    // ------------------------------------------------------------------------
    reg  [STATE_N_BITS - 1 : 0] state;
    reg  [STATE_N_BITS - 1 : 0] state_d;
    reg  [CNT_N_BITS - 1 : 0]   cnt;
    reg  [STEP_N_BITS - 1 : 0]  step;
    wire                        last_step;

    assign last_step = (step == N_STEPS);

    always @(*)
    begin
        case (state)
            STATE_IDLE:
            begin
                state_d = START ? STATE_RST : state;
            end

            STATE_RST:
            begin
                state_d = (cnt == RST_PW) ? STATE_RST_WAIT : state;
            end

            STATE_RST_WAIT:
            begin
                state_d = STATE_TX;
            end

            STATE_TX:
            begin
                state_d = (last_step && cnt == LAST_STEP_CYCLE) ? STATE_TX_WAIT : state;
            end

            STATE_TX_WAIT:
            begin
                state_d = (cnt == TX_WAIT) ? STATE_DONE : state;
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


    always @(posedge CLK)
    begin
        if (RESET)
        begin
            state <= STATE_IDLE;
            cnt <= 'd0;
            step <= 'd0;
        end
        else
        begin
            state <= state_d;

            if (state != state_d || state == STATE_IDLE || state == STATE_DONE)
            begin
                cnt <= 'd0;
                step <= 'd0;
            end
            else if (state == STATE_TX && cnt == LAST_STEP_CYCLE)
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


    // Datapath control
    integer         i;
    wire            bram_load_addr;
    wire            bram_burst_start;
    reg  [3 : 0]    bram_burst_chain;
    wire            step_sr_en;
    wire            step_buf_en;

    assign bram_load_addr = (state == STATE_IDLE) && START;

    assign bram_burst_start = !USE_RESET_CODE &&
                              ((state == STATE_RST && cnt == 'd0) ||
                               (state == STATE_RST_WAIT) ||
                               (state == STATE_TX && cnt == 'd3));

    assign step_sr_en = |bram_burst_chain[2 : 0];

    assign step_buf_en = bram_burst_chain[3];

    always @(posedge CLK)
    begin
        if (RESET)
        begin
            bram_burst_chain <= 'd0;
        end
        else
        begin
            bram_burst_chain[0] <= bram_burst_start;
            for (i = 1; i < 4; i = i + 1)
            begin
                bram_burst_chain[i] <= bram_burst_chain[i - 1];
            end
        end
    end


    // ------------------------------------------------------------------------
    // Datapath
    // ------------------------------------------------------------------------
    // BRAM interface
    wire [C_S_AXI_DATA_WIDTH / 2 - 1 : 0]   rx_bram_do;
    wire [C_S_AXI_DATA_WIDTH / 2 - 1 : 0]   tx_bram_do;

    assign BRAM_CLK = CLK;

    assign BRAM_EN = bram_burst_start || (|bram_burst_chain[1 : 0]);

    assign BRAM_WE = 'd0;

    assign BRAM_DI = 'd0;

    assign rx_bram_do = BRAM_DO[C_S_AXI_DATA_WIDTH / 2 - 1 : 0];

    assign tx_bram_do = BRAM_DO[C_S_AXI_DATA_WIDTH - 1 : C_S_AXI_DATA_WIDTH / 2];

    always @(posedge CLK)
    begin
        if (RESET)
        begin
            BRAM_ADDR <= 'd0;
        end
        else
        begin
            if (bram_load_addr)
            begin
                BRAM_ADDR <= {BUF_IDX, 9'd0};
            end
            else if (BRAM_EN)
            begin
                BRAM_ADDR <= BRAM_ADDR + 'd1;
            end
        end
    end


    // Buffers for mini-lvds frame
    reg  [FRAME_SIZE - 1 : 0]   rx_step_sr;
    reg  [FRAME_SIZE - 1 : 0]   tx_step_sr;
    reg  [FRAME_SIZE - 1 : 0]   rx_step_buf;
    reg  [FRAME_SIZE - 1 : 0]   tx_step_buf;
    wire [FRAME_SIZE - 1 : 0]   rx_step_buf_mux;
    wire [FRAME_SIZE - 1 : 0]   tx_step_buf_mux;

    always @(posedge CLK)
    begin
        if (RESET)
        begin
            rx_step_sr  <= 'd0;
            tx_step_sr  <= 'd0;
            rx_step_buf <= 'd0;
            tx_step_buf <= 'd0;
        end
        else
        begin
            if (step_sr_en)
            begin
                // BRAM read-out is LS byte first so shift into MS end
                rx_step_sr[47 : 32] <= rx_bram_do;
                rx_step_sr[31 : 16] <= rx_step_sr[47 : 32];
                rx_step_sr[15 :  0] <= rx_step_sr[31 : 16];

                tx_step_sr[47 : 32] <= tx_bram_do;
                tx_step_sr[31 : 16] <= tx_step_sr[47 : 32];
                tx_step_sr[15 :  0] <= tx_step_sr[31 : 16];
            end

            if (step_buf_en)
            begin
                rx_step_buf <= rx_step_sr;
                tx_step_buf <= tx_step_sr;
            end
        end
    end

    // The last LVDS frame is for Himax pins OUT1026, OUT1025,  ..., OUT1021.
    assign rx_step_buf_mux = last_step ?
                                {AUX_CODE_EVEN, AUX_CODE_ODD, rx_step_buf[31:0]} :
                                rx_step_buf;
    assign tx_step_buf_mux = last_step ?
                                {AUX_CODE_EVEN, AUX_CODE_ODD, tx_step_buf[31:0]} :
                                tx_step_buf;


    // Byte mux and bit mux
    genvar i_gv;
    wire [7 : 0] rx_bytes [5 : 0];
    wire [7 : 0] tx_bytes [5 : 0];
    wire [1 : 0] bit_sel;
    wire [2 : 0] bit_sel_d1;
    wire [2 : 0] bit_sel_d2;
    wire [5 : 0] rx_lvds_data_muxed_d1;
    wire [5 : 0] rx_lvds_data_muxed_d2;
    wire [5 : 0] tx_lvds_data_muxed_d1;
    wire [5 : 0] tx_lvds_data_muxed_d2;

    generate
        for (i_gv = 0; i_gv < 6; i_gv = i_gv + 1)
        begin : gen_bytes
            assign rx_bytes[i_gv] = USE_RESET_CODE ? RESET_CODE :
                                                     rx_step_buf_mux[8 * i_gv +: 8];
            assign tx_bytes[i_gv] = USE_RESET_CODE ? RESET_CODE :
                                                     tx_step_buf_mux[8 * i_gv +: 8];
        end
    endgenerate

    assign bit_sel = (state == STATE_TX) ? cnt[1 : 0] : 'd0;

    assign bit_sel_d1 = bit_sel << 1;

    assign bit_sel_d2 = bit_sel_d1 | 'd1;

    generate
        for (i_gv = 0; i_gv < 6; i_gv = i_gv + 1)
        begin : gen_bits
            assign rx_lvds_data_muxed_d1[i_gv] = rx_bytes[i_gv][bit_sel_d1];
            assign rx_lvds_data_muxed_d2[i_gv] = rx_bytes[i_gv][bit_sel_d2];
            assign tx_lvds_data_muxed_d1[i_gv] = tx_bytes[i_gv][bit_sel_d1];
            assign tx_lvds_data_muxed_d2[i_gv] = tx_bytes[i_gv][bit_sel_d2];
        end
    endgenerate


    // Output regs
    always @(posedge CLK)
    begin
        if (RESET)
        begin
            RX_LVDS_DATA_D1 <= 6'd0;
            RX_LVDS_DATA_D2 <= 6'd0;
            TX_LVDS_DATA_D1 <= 6'd0;
            TX_LVDS_DATA_D2 <= 6'd0;
        end
        else
        begin
            if (state == STATE_RST)
            begin
                // LVDS lane 0 is used for RST (other lanes are don't cares)
                RX_LVDS_DATA_D1 <= 6'b00000_1;
                RX_LVDS_DATA_D2 <= 6'b00000_1;
                TX_LVDS_DATA_D1 <= 6'b00000_1;
                TX_LVDS_DATA_D2 <= 6'b00000_1;
            end
            else if (state == STATE_TX)
            begin
                RX_LVDS_DATA_D1 <= rx_lvds_data_muxed_d1;
                RX_LVDS_DATA_D2 <= rx_lvds_data_muxed_d2;
                TX_LVDS_DATA_D1 <= tx_lvds_data_muxed_d1;
                TX_LVDS_DATA_D2 <= tx_lvds_data_muxed_d2;
            end
            else
            begin
                RX_LVDS_DATA_D1 <= 6'd0;
                RX_LVDS_DATA_D2 <= 6'd0;
                TX_LVDS_DATA_D1 <= 6'd0;
                TX_LVDS_DATA_D2 <= 6'd0;
            end
        end
    end

endmodule
