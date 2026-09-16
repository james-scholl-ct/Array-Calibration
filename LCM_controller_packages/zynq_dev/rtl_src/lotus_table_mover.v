`include "lotus_defines.v"
`timescale 1 ns / 1 ps

module lotus_table_mover #(
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 9,
    parameter integer ROM_DEPTH = 6528,
    parameter integer ROM_WIDTH = 32,
    parameter integer ROM_N_TABLES = 128
)
(
    input   wire                                        S_AXI_ACLK,
    input   wire                                        S_AXI_ARESETN,

    // Control
    input   wire                                        START_TXFER,
    input   wire    [$clog2(ROM_N_TABLES) - 1 : 0]      TABLE_IDX,
    output  wire                                        TXFER_BUSY,
    output  reg                                         TXFER_DONE,

    // RAM write interface (sink)
    input   wire                                        WR_ACK,
    output  wire                                        WR_REQ,
    output  wire    [C_S_AXI_ADDR_WIDTH - 1 : 0]        WR_ADDR,
    output  wire    [C_S_AXI_DATA_WIDTH - 1 : 0]        WR_DATA,
    output  wire    [(C_S_AXI_DATA_WIDTH / 8) - 1 : 0]  WR_STRB,

    // ROM read interface (source)
    input   wire    [ROM_WIDTH - 1  : 0]                RD_DATA,
    output  wire                                        RD_EN,
    output  wire    [$clog2(ROM_DEPTH) - 1 : 0]         RD_ADDR
);


    localparam integer BYTES_PER_WORD = C_S_AXI_DATA_WIDTH / 8;
    localparam integer WORD_ADDR_LSB = $clog2(BYTES_PER_WORD);
    localparam integer WORD_ADDR_N_BITS = C_S_AXI_ADDR_WIDTH - WORD_ADDR_LSB;

    localparam integer TABLE_N_WORDS = ROM_DEPTH / ROM_N_TABLES;

    localparam integer STATE_N_BITS = 2;
    localparam [STATE_N_BITS - 1 : 0] STATE_IDLE = 'd0;
    localparam [STATE_N_BITS - 1 : 0] STATE_READ = 'd1;
    localparam [STATE_N_BITS - 1 : 0] STATE_WRITE = 'd2;


    reg     [STATE_N_BITS - 1 : 0]          state;
    reg     [$clog2(ROM_DEPTH) - 1 : 0]     read_ptr;
    wire    [$clog2(ROM_DEPTH) - 1 : 0]     read_ptr_start;
    reg     [WORD_ADDR_N_BITS - 1 : 0]      write_ptr;
    wire    [WORD_ADDR_N_BITS - 1 : 0]      write_ptr_start;
    wire                                    done;

    assign read_ptr_start = TABLE_IDX * TABLE_N_WORDS;

    assign write_ptr_start = 'd0;

    assign done = (write_ptr == TABLE_N_WORDS - 1);

    assign TXFER_BUSY = (state != STATE_IDLE);

    always @( posedge S_AXI_ACLK )
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            state <= STATE_IDLE;
            read_ptr <= 'd0;
            write_ptr <= 'd0;
            TXFER_DONE <= 1'b0;
        end
        else
        begin
            TXFER_DONE <= done && WR_ACK;

            case( state )
                STATE_IDLE:
                begin
                    if( START_TXFER )
                    begin
                        state <= STATE_READ;
                        read_ptr <= read_ptr_start;
                        write_ptr <= write_ptr_start;
                    end
                end

                STATE_READ:
                begin
                    state <= STATE_WRITE;
                end

                STATE_WRITE:
                begin
                    if( WR_ACK )
                    begin
                        if( done )
                        begin
                            state <= STATE_IDLE;
                            read_ptr <= 'd0;
                            write_ptr <= 'd0;
                        end
                        else
                        begin
                            state <= STATE_READ;
                            read_ptr <= read_ptr + 'd1;
                            write_ptr <= write_ptr + 'd1;
                        end
                    end
                end

                default:
                begin
                    state <= STATE_IDLE;
                    read_ptr <= 'd0;
                    write_ptr <= 'd0;
                end
            endcase
        end
    end

    // ------------------------------------------------------------------------
    // Write
    // ------------------------------------------------------------------------
    assign WR_REQ = (state == STATE_WRITE) && !WR_ACK;

    assign WR_ADDR = write_ptr << WORD_ADDR_LSB;

    assign WR_DATA = RD_DATA;

    assign WR_STRB = {BYTES_PER_WORD{1'b1}};

    // ------------------------------------------------------------------------
    // Read
    // ------------------------------------------------------------------------
    assign RD_EN = (state == STATE_READ);

    assign RD_ADDR = read_ptr;

endmodule
