`include "lotus_defines.v"
`timescale 1 ns / 1 ps

module lotus_data_mux #(
    parameter integer STEP_N_BITS = 8,
    parameter integer TABLE_N_BITS = 32
)
(
    input   wire    [2 : 0]                             BIT_SEL,
    input   wire    [STEP_N_BITS - 1 : 0]               STEP,
    input   wire    [TABLE_N_BITS - 1 : 0]              TABLE,

    output  wire    [5 : 0]                             LVDS_DATA
);


    localparam integer TABLE_N_BYTES = TABLE_N_BITS / 8;

    // unpack the table into a byte array
    wire [7 : 0] bytes [TABLE_N_BYTES - 1 : 0];

    genvar i_gv;
    generate
        for( i_gv = 0; i_gv < TABLE_N_BYTES; i_gv = i_gv + 1 )
        begin : unpack
            assign bytes[i_gv] = TABLE[8 * i_gv +: 8];
        end
    endgenerate


    // mux bytes from the table and bits from each byte
    reg [7 : 0] b0;
    reg [7 : 0] b1;
    reg [7 : 0] b2;
    reg [7 : 0] b3;
    reg [7 : 0] b4;
    reg [7 : 0] b5;

    assign LVDS_DATA[0] = b0[BIT_SEL];
    assign LVDS_DATA[1] = b1[BIT_SEL];
    assign LVDS_DATA[2] = b2[BIT_SEL];
    assign LVDS_DATA[3] = b3[BIT_SEL];
    assign LVDS_DATA[4] = b4[BIT_SEL];
    assign LVDS_DATA[5] = b5[BIT_SEL];

    // STEP (0-151), b0-b5, byte (0-203), Bravo Rail (1-204)
    always @(*)
    begin
        case( STEP )
            8'd35:   b0 = bytes[70];    //   71
            8'd36:   b0 = bytes[76];    //   77
            8'd37:   b0 = bytes[80];    //   81
            8'd38:   b0 = bytes[82];    //   83
            8'd39:   b0 = bytes[90];    //   91
            8'd40:   b0 = bytes[68];    //   69
            8'd41:   b0 = bytes[62];    //   63
            8'd42:   b0 = bytes[64];    //   65
            8'd43:   b0 = bytes[46];    //   47
            8'd44:   b0 = bytes[52];    //   53
            8'd45:   b0 = bytes[18];    //   19
            8'd46:   b0 = bytes[54];    //   55
            8'd47:   b0 = bytes[48];    //   49
            8'd48:   b0 = bytes[125];   //  126
            8'd49:   b0 = bytes[147];   //  148
            8'd50:   b0 = bytes[139];   //  140
            8'd51:   b0 = bytes[106];   //  107
            8'd52:   b0 = bytes[149];   //  150
            8'd53:   b0 = bytes[143];   //  144
            8'd54:   b0 = bytes[120];   //  121
            8'd55:   b0 = bytes[153];   //  154
            8'd56:   b0 = bytes[108];   //  109
            8'd57:   b0 = bytes[116];   //  117
            8'd58:   b0 = bytes[115];   //  116
            8'd59:   b0 = bytes[156];   //  157
            8'd60:   b0 = bytes[130];   //  131
            8'd61:   b0 = bytes[152];   //  153
            8'd62:   b0 = bytes[137];   //  138
            8'd63:   b0 = bytes[129];   //  130
            8'd64:   b0 = bytes[162];   //  163
            8'd65:   b0 = bytes[93];    //   94
            8'd66:   b0 = bytes[122];   //  123
            8'd67:   b0 = bytes[95];    //   96
            8'd68:   b0 = bytes[87];    //   88
            8'd69:   b0 = bytes[180];   //  181
            8'd81:   b0 = bytes[182];   //  183
            8'd82:   b0 = bytes[77];    //   78
            8'd83:   b0 = bytes[146];   //  147
            8'd84:   b0 = bytes[79];    //   80
            8'd85:   b0 = bytes[71];    //   72
            8'd86:   b0 = bytes[191];   //  192
            8'd87:   b0 = bytes[61];    //   62
            8'd88:   b0 = bytes[178];   //  179
            8'd89:   b0 = bytes[67];    //   68
            8'd90:   b0 = bytes[55];    //   56
            8'd91:   b0 = bytes[198];   //  199
            8'd92:   b0 = bytes[33];    //   34
            8'd93:   b0 = bytes[193];   //  194
            8'd94:   b0 = bytes[41];    //   42
            8'd95:   b0 = bytes[29];    //   30
            8'd96:   b0 = bytes[177];   //  178
            8'd97:   b0 = bytes[43];    //   44
            8'd98:   b0 = bytes[174];   //  175
            8'd99:   b0 = bytes[13];    //   14
            8'd100:  b0 = bytes[9];     //   10
            8'd101:  b0 = bytes[23];    //   24
            8'd102:  b0 = bytes[27];    //   28
            8'd103:  b0 = bytes[6];     //    7
            8'd104:  b0 = bytes[26];    //   27
            8'd105:  b0 = bytes[42];    //   43
            8'd106:  b0 = bytes[14];    //   15
            8'd107:  b0 = bytes[32];    //   33
            8'd108:  b0 = bytes[22];    //   23
            8'd109:  b0 = bytes[24];    //   25
            8'd110:  b0 = bytes[12];    //   13
            8'd111:  b0 = bytes[16];    //   17
            8'd112:  b0 = bytes[4];     //    5
            8'd113:  b0 = bytes[8];     //    9
            8'd114:  b0 = bytes[10];    //   11
            8'd115:  b0 = bytes[7];     //    8
            default: b0 = 8'h00;
        endcase

        case( STEP )
            default: b1 = 8'h00;
        endcase

        case( STEP )
            8'd35:   b2 = bytes[167];   //  168
            8'd36:   b2 = bytes[86];    //   87
            8'd37:   b2 = bytes[84];    //   85
            8'd38:   b2 = bytes[88];    //   89
            8'd39:   b2 = bytes[94];    //   95
            8'd40:   b2 = bytes[78];    //   79
            8'd41:   b2 = bytes[50];    //   51
            8'd42:   b2 = bytes[60];    //   61
            8'd43:   b2 = bytes[34];    //   35
            8'd44:   b2 = bytes[58];    //   59
            8'd45:   b2 = bytes[56];    //   57
            8'd46:   b2 = bytes[38];    //   39
            8'd47:   b2 = bytes[44];    //   45
            8'd48:   b2 = bytes[171];   //  172
            8'd49:   b2 = bytes[155];   //  156
            8'd50:   b2 = bytes[131];   //  132
            8'd51:   b2 = bytes[127];   //  128
            8'd52:   b2 = bytes[150];   //  151
            8'd53:   b2 = bytes[135];   //  136
            8'd54:   b2 = bytes[112];   //  113
            8'd55:   b2 = bytes[145];   //  146
            8'd56:   b2 = bytes[136];   //  137
            8'd57:   b2 = bytes[104];   //  105
            8'd58:   b2 = bytes[113];   //  114
            8'd59:   b2 = bytes[144];   //  145
            8'd60:   b2 = bytes[111];   //  112
            8'd61:   b2 = bytes[105];   //  106
            8'd62:   b2 = bytes[114];   //  115
            8'd63:   b2 = bytes[107];   //  108
            8'd64:   b2 = bytes[160];   //  161
            8'd65:   b2 = bytes[109];   //  110
            8'd66:   b2 = bytes[97];    //   98
            8'd67:   b2 = bytes[170];   //  171
            8'd68:   b2 = bytes[91];    //   92
            8'd69:   b2 = bytes[134];   //  135
            8'd81:   b2 = bytes[176];   //  177
            8'd82:   b2 = bytes[89];    //   90
            8'd83:   b2 = bytes[73];    //   74
            8'd84:   b2 = bytes[196];   //  197
            8'd85:   b2 = bytes[75];    //   76
            8'd86:   b2 = bytes[199];   //  200
            8'd87:   b2 = bytes[65];    //   66
            8'd88:   b2 = bytes[49];    //   50
            8'd89:   b2 = bytes[187];   //  188
            8'd90:   b2 = bytes[53];    //   54
            8'd91:   b2 = bytes[158];   //  159
            8'd92:   b2 = bytes[59];    //   60
            8'd93:   b2 = bytes[51];    //   52
            8'd94:   b2 = bytes[202];   //  203
            8'd95:   b2 = bytes[25];    //   26
            8'd96:   b2 = bytes[185];   //  186
            8'd97:   b2 = bytes[39];    //   40
            8'd98:   b2 = bytes[31];    //   32
            8'd99:   b2 = bytes[157];   //  158
            8'd100:  b2 = bytes[5];     //    6
            8'd101:  b2 = bytes[11];    //   12
            8'd102:  b2 = bytes[19];    //   20
            8'd103:  b2 = bytes[36];    //   37
            8'd104:  b2 = bytes[40];    //   41
            8'd105:  b2 = bytes[30];    //   31
            8'd106:  b2 = bytes[28];    //   29
            8'd107:  b2 = bytes[2];     //    3
            8'd108:  b2 = bytes[20];    //   21
            8'd109:  b2 = bytes[0];     //    1
            default: b2 = 8'h00;
        endcase

        case( STEP )
            default: b3 = 8'h00;
        endcase

        case( STEP )
            8'd35:   b4 = bytes[163];   //  164
            8'd36:   b4 = bytes[98];    //   99
            8'd37:   b4 = bytes[102];   //  103
            8'd38:   b4 = bytes[92];    //   93
            8'd39:   b4 = bytes[96];    //   97
            8'd40:   b4 = bytes[74];    //   75
            8'd41:   b4 = bytes[72];    //   73
            8'd42:   b4 = bytes[66];    //   67
            8'd43:   b4 = bytes[132];   //  133
            8'd44:   b4 = bytes[164];   //  165
            8'd45:   b4 = bytes[142];   //  143
            8'd46:   b4 = bytes[172];   //  173
            8'd47:   b4 = bytes[154];   //  155
            8'd48:   b4 = bytes[159];   //  160
            8'd49:   b4 = bytes[151];   //  152
            8'd50:   b4 = bytes[110];   //  111
            8'd51:   b4 = bytes[161];   //  162
            8'd52:   b4 = bytes[128];   //  129
            8'd53:   b4 = bytes[123];   //  124
            8'd54:   b4 = bytes[119];   //  120
            8'd55:   b4 = bytes[100];   //  101
            8'd56:   b4 = bytes[124];   //  125
            8'd57:   b4 = bytes[148];   //  149
            8'd58:   b4 = bytes[133];   //  134
            8'd59:   b4 = bytes[138];   //  139
            8'd60:   b4 = bytes[140];   //  141
            8'd61:   b4 = bytes[121];   //  122
            8'd62:   b4 = bytes[126];   //  127
            8'd63:   b4 = bytes[103];   //  104
            8'd64:   b4 = bytes[99];    //  100
            8'd65:   b4 = bytes[118];   //  119
            8'd66:   b4 = bytes[117];   //  118
            8'd67:   b4 = bytes[168];   //  169
            8'd68:   b4 = bytes[81];    //   82
            8'd69:   b4 = bytes[175];   //  176
            8'd80:   b4 = bytes[101];   //  102
            8'd81:   b4 = bytes[83];    //   84
            8'd82:   b4 = bytes[188];   //  189
            8'd83:   b4 = bytes[85];    //   86
            8'd84:   b4 = bytes[194];   //  195
            8'd85:   b4 = bytes[69];    //   70
            8'd86:   b4 = bytes[57];    //   58
            8'd87:   b4 = bytes[192];   //  193
            8'd88:   b4 = bytes[63];    //   64
            8'd89:   b4 = bytes[195];   //  196
            8'd90:   b4 = bytes[45];    //   46
            8'd91:   b4 = bytes[37];    //   38
            8'd92:   b4 = bytes[181];   //  182
            8'd93:   b4 = bytes[47];    //   48
            8'd94:   b4 = bytes[186];   //  187
            8'd95:   b4 = bytes[17];    //   18
            8'd96:   b4 = bytes[1];     //    2
            8'd97:   b4 = bytes[190];   //  191
            8'd98:   b4 = bytes[21];    //   22
            8'd99:   b4 = bytes[173];   //  174
            8'd100:  b4 = bytes[35];    //   36
            8'd101:  b4 = bytes[3];     //    4
            8'd102:  b4 = bytes[166];   //  167
            8'd103:  b4 = bytes[184];   //  185
            8'd104:  b4 = bytes[203];   //  204
            8'd105:  b4 = bytes[183];   //  184
            8'd106:  b4 = bytes[200];   //  201
            8'd107:  b4 = bytes[179];   //  180
            8'd108:  b4 = bytes[197];   //  198
            8'd109:  b4 = bytes[169];   //  170
            8'd110:  b4 = bytes[201];   //  202
            8'd111:  b4 = bytes[165];   //  166
            8'd112:  b4 = bytes[189];   //  190
            8'd113:  b4 = bytes[141];   //  142
            8'd114:  b4 = bytes[15];    //   16
            default: b4 = 8'h00;
        endcase

        case( STEP )
            default: b5 = 8'h00;
        endcase
    end

endmodule
