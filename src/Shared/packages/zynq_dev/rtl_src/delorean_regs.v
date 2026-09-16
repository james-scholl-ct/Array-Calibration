`include "delorean_defines.v"
`timescale 1 ns / 1 ps

module delorean_regs #(
    parameter integer C_S_AXI_DATA_DEPTH = 32,
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer C_S_AXI_ADDR_WIDTH = 12,
    parameter integer N_STATUS_WORDS = 1
)
(
    // Global signals
    input   wire                                        S_AXI_ACLK,
    input   wire                                        S_AXI_ARESETN,

    // Upstream write interfaces
    input   wire                                        CH0_WR_REQ,
    input   wire    [C_S_AXI_ADDR_WIDTH - 1 : 0]        CH0_WR_ADDR,
    input   wire    [C_S_AXI_DATA_WIDTH - 1 : 0]        CH0_WR_DATA,
    input   wire    [(C_S_AXI_DATA_WIDTH / 8) - 1 : 0]  CH0_WR_STRB,
    output  reg                                         CH0_WR_ACK,

    input   wire                                        CH1_WR_REQ,
    input   wire    [C_S_AXI_ADDR_WIDTH - 1 : 0]        CH1_WR_ADDR,
    input   wire    [C_S_AXI_DATA_WIDTH - 1 : 0]        CH1_WR_DATA,
    input   wire    [(C_S_AXI_DATA_WIDTH / 8) - 1 : 0]  CH1_WR_STRB,
    output  reg                                         CH1_WR_ACK,

    // Upstream read interfaces
    input   wire                                        CH0_RD_EN,
    input   wire    [C_S_AXI_ADDR_WIDTH - 1 : 0]        CH0_RD_ADDR,
    output  wire                                        CH0_RD_VALID,
    output  wire    [C_S_AXI_DATA_WIDTH - 1  : 0]       CH0_RD_DATA,

    // Downstream memory interface (as raw memory in lieu of read interface)
    input   wire    [N_STATUS_WORDS*C_S_AXI_DATA_WIDTH - 1 : 0] STATUS,
    input   wire                                        WR_BACK_PRESSURE,
    output  wire    [C_S_AXI_DATA_DEPTH*C_S_AXI_DATA_WIDTH - 1 : 0] MEMORY,
    output  wire    [C_S_AXI_DATA_DEPTH - 1 : 0]        READ_VALIDS,
    output  reg     [C_S_AXI_DATA_DEPTH - 1 : 0]        WRITE_VALIDS
);


    localparam integer BYTES_PER_WORD = C_S_AXI_DATA_WIDTH / 8;
    localparam integer WORD_ADDR_LSB = $clog2(BYTES_PER_WORD);
    localparam integer WORD_ADDR_N_BITS = C_S_AXI_ADDR_WIDTH - WORD_ADDR_LSB;

    localparam [WORD_ADDR_N_BITS - 1 : 0] STATUS_OFFSET = C_S_AXI_DATA_DEPTH - N_STATUS_WORDS;

    reg  [C_S_AXI_DATA_WIDTH - 1 : 0] regfile [C_S_AXI_DATA_DEPTH - 1 : 0];
    wire [C_S_AXI_DATA_WIDTH - 1 : 0] status_u [N_STATUS_WORDS - 1 : 0];

    // ------------------------------------------------------------------------
    // Write arbitration
    // ------------------------------------------------------------------------
    reg                                         write_en;
    reg     [C_S_AXI_ADDR_WIDTH - 1 : 0]        write_addr_pp;
    reg     [C_S_AXI_DATA_WIDTH - 1  : 0]       write_data_pp;
    reg     [(C_S_AXI_DATA_WIDTH / 8) - 1 : 0]  write_strb_pp;

    always @( posedge S_AXI_ACLK )
    begin
        if ( S_AXI_ARESETN == 1'b0 )
        begin
            write_en      <= 1'b0;
            CH0_WR_ACK    <= 1'b0;
            CH1_WR_ACK    <= 1'b0;
            write_addr_pp <= 'd0;
            write_data_pp <= 'd0;
            write_strb_pp <= 'd0;
        end
        else
        begin
            if( !WR_BACK_PRESSURE && CH0_WR_REQ )
            begin
                CH0_WR_ACK    <= 1'b1;
                CH1_WR_ACK    <= 1'b0;
                write_en      <= 1'b1;
                write_addr_pp <= CH0_WR_ADDR;
                write_data_pp <= CH0_WR_DATA;
                write_strb_pp <= CH0_WR_STRB;
            end
            else if( !WR_BACK_PRESSURE && CH1_WR_REQ )
            begin
                CH0_WR_ACK    <= 1'b0;
                CH1_WR_ACK    <= 1'b1;
                write_en      <= 1'b1;
                write_addr_pp <= CH1_WR_ADDR;
                write_data_pp <= CH1_WR_DATA;
                write_strb_pp <= CH1_WR_STRB;
            end
            else
            begin
                CH0_WR_ACK    <= 1'b0;
                CH1_WR_ACK    <= 1'b0;
                write_en      <= 1'b0;
                write_addr_pp <= write_addr_pp;
                write_data_pp <= write_data_pp;
                write_strb_pp <= write_strb_pp;
            end
        end
    end

    // ------------------------------------------------------------------------
    // Write
    // ------------------------------------------------------------------------
    wire    [WORD_ADDR_N_BITS - 1 : 0]      wr_word_addr;
    wire    [C_S_AXI_DATA_DEPTH - 1 : 0]    write_enables;

    assign wr_word_addr = write_addr_pp[C_S_AXI_ADDR_WIDTH - 1 : WORD_ADDR_LSB];

    assign write_enables = write_en << wr_word_addr;

    integer i_word;
    integer i_byte;

    always @( posedge S_AXI_ACLK )
    begin
        if( !S_AXI_ARESETN )
        begin
            WRITE_VALIDS <= 'd0;

            for( i_word = 0; i_word < C_S_AXI_DATA_DEPTH; i_word = i_word + 1 )
            begin
                regfile[i_word] <= 'd0;
            end
        end
        else
        begin
            WRITE_VALIDS <= write_enables;

            for( i_word = 0; i_word < C_S_AXI_DATA_DEPTH; i_word = i_word + 1 )
            begin
                if( write_enables[i_word] )
                begin
                    for( i_byte = 0; i_byte < BYTES_PER_WORD; i_byte = i_byte + 1 )
                    begin
                        if( write_strb_pp[i_byte] )
                        begin
                            regfile[i_word][(i_byte*8) +: 8] <=
                              write_data_pp[(i_byte*8) +: 8];
                        end
                    end
                end
            end
        end
    end

    // ------------------------------------------------------------------------
    // Read
    // ------------------------------------------------------------------------
    reg                                     ch0_read_valid;
    reg     [WORD_ADDR_N_BITS - 1 : 0]      ch0_read_word_addr;
    wire    [WORD_ADDR_N_BITS - 1 : 0]      ch0_read_word_addr_status;
    wire                                    ch0_read_status;

    always @( posedge S_AXI_ACLK )
    begin
        if( S_AXI_ARESETN == 1'b0 )
        begin
            ch0_read_valid     <= 1'b0;
            ch0_read_word_addr <= 'd0;
        end
        else
        begin
            ch0_read_valid <= CH0_RD_EN;
            if( CH0_RD_EN )
                ch0_read_word_addr <= CH0_RD_ADDR[C_S_AXI_ADDR_WIDTH - 1 : WORD_ADDR_LSB];
        end
    end

    assign CH0_RD_VALID = ch0_read_valid;

    assign ch0_read_status = (ch0_read_word_addr >= STATUS_OFFSET);

    assign ch0_read_word_addr_status = ch0_read_word_addr - STATUS_OFFSET;

    assign CH0_RD_DATA = ch0_read_status ? status_u[ch0_read_word_addr_status] :
                                           regfile[ch0_read_word_addr];

    assign READ_VALIDS = (ch0_read_valid << ch0_read_word_addr);

    // ------------------------------------------------------------------------
    // Memory pack/unpack
    // ------------------------------------------------------------------------
    genvar i_gv;
    generate
        for( i_gv = 0; i_gv < C_S_AXI_DATA_DEPTH; i_gv = i_gv + 1 )
        begin : pack
            assign MEMORY[C_S_AXI_DATA_WIDTH * (i_gv+1) - 1 :
                          C_S_AXI_DATA_WIDTH * i_gv         ] = regfile[i_gv];
        end

        for( i_gv = 0; i_gv < N_STATUS_WORDS; i_gv = i_gv + 1 )
        begin : unpack
            assign status_u[i_gv] = STATUS[C_S_AXI_DATA_WIDTH * (i_gv+1) - 1 :
                                           C_S_AXI_DATA_WIDTH * i_gv         ];
        end
    endgenerate

endmodule
