`include "delorean_defines.v"
`timescale 1 ns / 1 ps

module delorean_spi_fsm #(
    // Config / Control
    parameter integer CONTROL_N_BITS = 32,
    parameter integer CLK_CONFIG_N_BITS = 32,
    parameter integer TABLE_N_BITS = 32,
    // FIFO side
    parameter integer CMD_FIFO_WIDTH = 32,
    parameter integer RSP_FIFO_WIDTH = 32,
    parameter integer RSP_FIFO_DEPTH = 32,
    // Controller side
    parameter integer BASE_FRAME_N_BITS = 16,
    parameter integer CLK_DIV_N_BITS = 4,
    parameter integer MAX_BASE_FRAMES = 16,
    parameter integer N_SLAVES = 8
)
(
    input   wire                                        S_AXI_ACLK,
    input   wire                                        S_AXI_ARESETN,

    // Config / Control
    input   wire    [CONTROL_N_BITS - 1 : 0]            SPI_CONTROL,
    input   wire    [CLK_CONFIG_N_BITS - 1 : 0]         SPI_CLK_CONFIG,
    input   wire    [TABLE_N_BITS - 1 : 0]              SPI_TABLE,

    // FIFO side
    input   wire                                        CMD_FIFO_EMPTY,
    input   wire    [CMD_FIFO_WIDTH - 1 : 0]            CMD_FIFO_RDATA,
    output  wire                                        CMD_FIFO_REN,

    input   wire                                        RSP_FIFO_FULL_0,
    input   wire                                        RSP_FIFO_FULL_1,
    output  reg     [RSP_FIFO_WIDTH - 1 : 0]            RSP_FIFO_WDATA_0,
    output  reg     [RSP_FIFO_WIDTH - 1 : 0]            RSP_FIFO_WDATA_1,
    output  reg                                         RSP_FIFO_WEN_0,
    output  reg                                         RSP_FIFO_WEN_1,

    // Controller side
    input   wire                                        DONE,
    input   wire    [BASE_FRAME_N_BITS - 1 : 0]         RDATA_0,
    input   wire    [BASE_FRAME_N_BITS - 1 : 0]         RDATA_1,
    input   wire                                        WDATA_REQ,

    output  wire    [1 : 0]                             ADC_CHSEL,
    output  wire    [CLK_DIV_N_BITS - 1 : 0]            CLK_DIV,
    output  wire                                        DAISY_EN,
    output  reg     [$clog2(MAX_BASE_FRAMES) - 1 : 0]   N_BASE_FRAMES, // 0-15 -> 1-16
    output  wire    [$clog2(N_SLAVES) - 1 : 0]          SLAVE_IDX,
    output  wire                                        START,
    output  reg     [BASE_FRAME_N_BITS - 1 : 0]         WDATA_0,
    output  reg     [BASE_FRAME_N_BITS - 1 : 0]         WDATA_1
);


    localparam integer IDX_FILL_CMD_FIFO = 0;
    localparam integer IDX_DAISY_EN = 1;
    localparam integer IDX_ADC_CHSEL = 2;

    localparam integer STATE_N_BITS = 2;
    localparam [STATE_N_BITS - 1 : 0] STATE_IDLE = 'd0;
    localparam [STATE_N_BITS - 1 : 0] STATE_START = 'd1;
    localparam [STATE_N_BITS - 1 : 0] STATE_RUN = 'd2;
    localparam [STATE_N_BITS - 1 : 0] STATE_WAIT = 'd3;

    localparam integer OPCODE_N_BITS = 4;
    localparam [OPCODE_N_BITS - 1 : 0] OPCODE_STANDARD = 'd0;
    localparam [OPCODE_N_BITS - 1 : 0] OPCODE_JUMBO = 'd1;
    localparam [OPCODE_N_BITS - 1 : 0] OPCODE_STREAM = 'd2;

    localparam integer CMD_PAYLOAD_N_BITS = CMD_FIFO_WIDTH -
                                            OPCODE_N_BITS -
                                            1 -
                                            $clog2(N_SLAVES);
    localparam integer FRAME_CNT_N_BITS = $clog2(RSP_FIFO_DEPTH);
    localparam integer CNT_N_BITS = CMD_PAYLOAD_N_BITS - FRAME_CNT_N_BITS;

    // ------------------------------------------------------------------------
    // Control and Config
    // ------------------------------------------------------------------------
    wire                                fill_cmd_fifo;
    wire [BASE_FRAME_N_BITS - 1 : 0]    frame_data_0[MAX_BASE_FRAMES - 1 : 0];
    wire [BASE_FRAME_N_BITS - 1 : 0]    frame_data_1[MAX_BASE_FRAMES - 1 : 0];

    assign DAISY_EN = SPI_CONTROL[IDX_DAISY_EN];

    assign fill_cmd_fifo = SPI_CONTROL[IDX_FILL_CMD_FIFO];

    assign ADC_CHSEL = SPI_CONTROL[IDX_ADC_CHSEL +: 2];

    assign CLK_DIV = SPI_CLK_CONFIG[SLAVE_IDX * CLK_DIV_N_BITS +: CLK_DIV_N_BITS];

    genvar i_gv;
    generate
        for( i_gv = 0; i_gv < MAX_BASE_FRAMES; i_gv = i_gv + 1 )
        begin : unpack
            assign frame_data_0[i_gv] = SPI_TABLE[BASE_FRAME_N_BITS * i_gv +:
                                                  BASE_FRAME_N_BITS];

            assign frame_data_1[i_gv] = SPI_TABLE[BASE_FRAME_N_BITS * (i_gv + MAX_BASE_FRAMES) +:
                                                  BASE_FRAME_N_BITS];
        end
    endgenerate

    // ------------------------------------------------------------------------
    // State machine
    // ------------------------------------------------------------------------
    reg     [STATE_N_BITS - 1 : 0]              state;
    reg     [CMD_FIFO_WIDTH - 1 : 0]            cmd;
    reg     [$clog2(MAX_BASE_FRAMES) - 1 : 0]   frame_ptr;
    reg     [CNT_N_BITS - 1 : 0]                cnt;
    reg     [FRAME_CNT_N_BITS - 1 : 0]          frame_cnt;
    wire    [OPCODE_N_BITS - 1 : 0]             opcode;
    wire                                        rfu;
    wire    [CMD_PAYLOAD_N_BITS - 1 : 0]        cmd_payload;
    reg     [FRAME_CNT_N_BITS - 1 : 0]          n_frames;
    reg     [CNT_N_BITS - 1 : 0]                n_wait_cycles;

    assign CMD_FIFO_REN = (state == STATE_IDLE) &&
                          !CMD_FIFO_EMPTY &&
                          !fill_cmd_fifo;

    assign START = (state == STATE_START);

    assign {opcode, rfu, SLAVE_IDX, cmd_payload} = cmd;

    always @( posedge S_AXI_ACLK )
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            state <= STATE_IDLE;
            cmd <= 'd0;
            frame_ptr <= 'd0;
            cnt <= 'd0;
            frame_cnt <= 'd0;
        end
        else
        begin
            case( state )
                STATE_IDLE:
                begin
                    if( CMD_FIFO_REN )
                    begin
                        state <= STATE_START;
                        cmd <= CMD_FIFO_RDATA;
                    end
                    else
                    begin
                        state <= STATE_IDLE;
                        cmd <= 'd0;
                        frame_ptr <= 'd0;
                        cnt <= 'd0;
                        frame_cnt <= 'd0;
                    end
                end

                STATE_START:
                begin
                    state <= STATE_RUN;
                end

                STATE_RUN:
                begin
                    if( DONE )
                    begin
                        state <= STATE_WAIT;
                        frame_ptr <= 'd0;
                    end
                    else if( WDATA_REQ )
                    begin
                        frame_ptr <= frame_ptr + 'd1;
                    end
                end

                STATE_WAIT:
                begin
                    if( cnt == n_wait_cycles )
                    begin
                        cnt <= 'd0;
                        if( frame_cnt == n_frames )
                        begin
                            state <= STATE_IDLE;
                            frame_cnt <= 'd0;
                        end
                        else
                        begin
                            state <= STATE_START;
                            frame_cnt <= frame_cnt + 'd1;
                        end
                    end
                    else
                    begin
                        cnt <= cnt + 'd1;
                    end
                end

                default:
                begin
                    state <= STATE_IDLE;
                    cmd <= 'd0;
                    frame_ptr <= 'd0;
                    cnt <= 'd0;
                    frame_cnt <= 'd0;
                end
            endcase
        end
    end


    always @(*)
    begin
        case(opcode)
            OPCODE_STANDARD:
            begin
                WDATA_0 = cmd_payload[BASE_FRAME_N_BITS - 1 : 0];
                WDATA_1 = cmd_payload[BASE_FRAME_N_BITS - 1 : 0];
                N_BASE_FRAMES = 'd0;
                n_frames = 'd0;
                n_wait_cycles = 'd0;
                RSP_FIFO_WEN_0 = !RSP_FIFO_FULL_0 && DONE;
                RSP_FIFO_WEN_1 = !RSP_FIFO_FULL_1 && DONE;
                RSP_FIFO_WDATA_0 = {SLAVE_IDX, RDATA_0};
                RSP_FIFO_WDATA_1 = {SLAVE_IDX, RDATA_1};
            end

            OPCODE_JUMBO:
            begin
                WDATA_0 = frame_data_0[frame_ptr];
                WDATA_1 = frame_data_1[frame_ptr];
                N_BASE_FRAMES = cmd_payload[$clog2(MAX_BASE_FRAMES) - 1 : 0];
                n_frames = 'd0;
                n_wait_cycles = 'd0;
                RSP_FIFO_WEN_0 = 1'b0;
                RSP_FIFO_WEN_1 = 1'b0;
                RSP_FIFO_WDATA_0 = 'd0;
                RSP_FIFO_WDATA_1 = 'd0;
            end

            OPCODE_STREAM:
            begin
                WDATA_0 = 'd0;
                WDATA_1 = 'd0;
                N_BASE_FRAMES = 'd0;
                n_frames = cmd_payload[FRAME_CNT_N_BITS - 1 : 0];
                n_wait_cycles = cmd_payload[CMD_PAYLOAD_N_BITS - 1 : FRAME_CNT_N_BITS];
                RSP_FIFO_WEN_0 = !RSP_FIFO_FULL_0 && DONE;
                RSP_FIFO_WEN_1 = !RSP_FIFO_FULL_1 && DONE;
                RSP_FIFO_WDATA_0 = {SLAVE_IDX, RDATA_0};
                RSP_FIFO_WDATA_1 = {SLAVE_IDX, RDATA_1};
            end

            default:
            begin
                WDATA_0 = 'd0;
                WDATA_1 = 'd0;
                N_BASE_FRAMES = 'd0;
                n_frames = 'd0;
                n_wait_cycles = 'd0;
                RSP_FIFO_WEN_0 = 1'b0;
                RSP_FIFO_WEN_1 = 1'b0;
                RSP_FIFO_WDATA_0 = 'd0;
                RSP_FIFO_WDATA_1 = 'd0;
            end
        endcase
    end

endmodule
