`timescale 1 ns / 1 ps

module orchid_data_mux #(
    parameter integer C_S_AXI_DATA_DEPTH = 72,
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    parameter integer N_DRIVE_CHANNELS   = 64
)
(
    input       [C_S_AXI_DATA_DEPTH*C_S_AXI_DATA_WIDTH-1 : 0 ] AXI_MEMORY,
    input       [ 2 : 0 ]                       BIT_SEL,
    input                                       CLK,
    input                                       FORCE_ZEROS,
    input                                       RESET,
    input                                       RST_EN,
    input       [ 7 : 0 ]                       STEP,
    input       [ 1 : 0 ]                       TABLE_SEL,
    input                                       TX_EN,

    output reg  [ 5 : 0 ]                       LVDS_DATA
);


    localparam integer N_TABLES = 4;
    localparam integer TABLE_N_WORDS = 8 * N_DRIVE_CHANNELS / C_S_AXI_DATA_WIDTH;
    localparam integer WORD_N_BYTES = C_S_AXI_DATA_WIDTH / 8;


    // unpack axi memory
    wire [C_S_AXI_DATA_WIDTH - 1 : 0] axi_memory_u [C_S_AXI_DATA_DEPTH - 1 : 0];

    genvar i_gv;
    generate
        for( i_gv = 0; i_gv < C_S_AXI_DATA_DEPTH; i_gv = i_gv + 1 )
        begin : array_reshape
            assign axi_memory_u[ i_gv ] =
                AXI_MEMORY[ C_S_AXI_DATA_WIDTH * (i_gv+1) - 1 : C_S_AXI_DATA_WIDTH * i_gv ];
        end
    endgenerate


    // select desired table
    wire [C_S_AXI_DATA_WIDTH - 1 : 0] tables [N_TABLES - 1 : 0][TABLE_N_WORDS - 1 : 0];
    wire [C_S_AXI_DATA_WIDTH - 1 : 0] table_muxed [TABLE_N_WORDS - 1 : 0];

    genvar j_gv;
    generate
        for( i_gv = 0; i_gv < N_TABLES; i_gv = i_gv + 1 )
        begin : tables_i_gv
            for( j_gv = 0; j_gv < TABLE_N_WORDS; j_gv = j_gv + 1 )
            begin : tables_j_gv
                assign tables[i_gv][j_gv] = axi_memory_u[i_gv*TABLE_N_WORDS + j_gv];
                // the mux
                assign table_muxed[j_gv] = tables[TABLE_SEL][j_gv];
            end
        end
    endgenerate


    // reshape muxed table into byte array
    wire [ 7 : 0 ] bytes [N_DRIVE_CHANNELS - 1 : 0];

    generate
        for( i_gv = 0; i_gv < TABLE_N_WORDS; i_gv = i_gv + 1 )
        begin : bytes_i_gv
            for( j_gv = 0; j_gv < WORD_N_BYTES; j_gv = j_gv + 1 )
            begin : bytes_j_gv
                assign bytes[i_gv*WORD_N_BYTES + j_gv] = table_muxed[i_gv][8*(j_gv+1) - 1 : 8*j_gv];
            end
        end
    endgenerate


    // select the byte for each of the 6 LVDS channels
    reg [ 7 : 0 ] b0;
    reg [ 7 : 0 ] b1;
    reg [ 7 : 0 ] b2;
    reg [ 7 : 0 ] b3;
    reg [ 7 : 0 ] b4;
    reg [ 7 : 0 ] b5;

    always @(*)
    begin
        case( STEP )
            8'd0:	 b0 = bytes[32];
            8'd8:	 b0 = bytes[33];
            8'd16:	 b0 = bytes[34];
            8'd24:	 b0 = bytes[35];
            8'd32:	 b0 = bytes[36];
            8'd40:	 b0 = bytes[39];
            8'd44:	 b0 = bytes[41];
            8'd48:	 b0 = bytes[42];
            8'd56:	 b0 = bytes[43];
            8'd64:	 b0 = bytes[44];
            8'd72:	 b0 = bytes[45];
            8'd80:	 b0 = bytes[46];
            8'd125:	 b0 = bytes[52];
            8'd131:	 b0 = bytes[56];
            default: b0 = 8'h00;
        endcase

        case( STEP )
            8'd0:	 b1 = bytes[0];
            8'd8:	 b1 = bytes[1];
            8'd16:	 b1 = bytes[2];
            8'd24:	 b1 = bytes[3];
            8'd32:	 b1 = bytes[4];
            8'd39:	 b1 = bytes[5];
            8'd40:	 b1 = bytes[8];
            8'd44:	 b1 = bytes[11];
            8'd48:	 b1 = bytes[12];
            8'd56:	 b1 = bytes[13];
            8'd64:	 b1 = bytes[14];
            8'd72:	 b1 = bytes[15];
            8'd80:	 b1 = bytes[16];
            8'd125:	 b1 = bytes[22];
            8'd131:	 b1 = bytes[25];
            default: b1 = 8'h00;
        endcase

        case( STEP )
            8'd39:	 b2 = bytes[37];
            8'd125:	 b2 = bytes[53];
            8'd131:	 b2 = bytes[57];
            default: b2 = 8'h00;
        endcase

        case( STEP )
            8'd39:	 b3 = bytes[6];
            8'd43:	 b3 = bytes[9];
            8'd125:	 b3 = bytes[23];
            8'd131:	 b3 = bytes[26];
            default: b3 = 8'h00;
        endcase

        case( STEP )
            8'd39:	 b4 = bytes[38];
            8'd43:	 b4 = bytes[40];
            8'd90:	 b4 = bytes[47];
            8'd98:	 b4 = bytes[48];
            8'd106:	 b4 = bytes[49];
            8'd114:	 b4 = bytes[50];
            8'd122:	 b4 = bytes[51];
            8'd125:	 b4 = bytes[54];
            8'd130:	 b4 = bytes[55];
            8'd131:	 b4 = bytes[58];
            8'd138:	 b4 = bytes[59];
            8'd146:	 b4 = bytes[60];
            8'd154:	 b4 = bytes[61];
            8'd162:	 b4 = bytes[62];
            8'd170:	 b4 = bytes[63];
            default: b4 = 8'h00;
        endcase

        case( STEP )
            8'd39:	 b5 = bytes[7];
            8'd43:	 b5 = bytes[10];
            8'd90:	 b5 = bytes[17];
            8'd98:	 b5 = bytes[18];
            8'd106:	 b5 = bytes[19];
            8'd114:	 b5 = bytes[20];
            8'd122:	 b5 = bytes[21];
            8'd130:	 b5 = bytes[24];
            8'd138:	 b5 = bytes[27];
            8'd146:	 b5 = bytes[28];
            8'd154:	 b5 = bytes[29];
            8'd162:	 b5 = bytes[30];
            8'd170:	 b5 = bytes[31];
            default: b5 = 8'h00;
        endcase
    end


    // select the bit in the byte for each of the 6 LVDS channels
    wire [ 5 : 0 ] lvds_data_muxed;

    assign lvds_data_muxed[0] = b0[BIT_SEL];
    assign lvds_data_muxed[1] = b1[BIT_SEL];
    assign lvds_data_muxed[2] = b2[BIT_SEL];
    assign lvds_data_muxed[3] = b3[BIT_SEL];
    assign lvds_data_muxed[4] = b4[BIT_SEL];
    assign lvds_data_muxed[5] = b5[BIT_SEL];


    // register outputs
    always @( posedge CLK )
    begin
        if( RESET )
        begin
            LVDS_DATA <= 6'd0;
        end
        else
        begin
            LVDS_DATA[0]   <= RST_EN      ? 1'b1 :
                              FORCE_ZEROS ? 1'b0 :
                              TX_EN       ? lvds_data_muxed[0] :
                                            1'b0;

            LVDS_DATA[5:1] <= RST_EN      ? 5'b0 :
                              FORCE_ZEROS ? 5'b0 :
                              TX_EN       ? lvds_data_muxed[5:1] :
                                            5'b0;
        end
    end


endmodule
