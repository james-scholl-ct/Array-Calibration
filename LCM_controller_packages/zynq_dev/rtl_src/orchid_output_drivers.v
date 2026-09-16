
`timescale 1 ns / 1 ps

module orchid_output_drivers (
    input       LVDS_0,
    input       LVDS_1,
    input       LVDS_2,
    input       LVDS_3,
    input       LVDS_4,
    input       LVDS_5,
    input       LVDS_CLK,
    input       OEN,
    input       POL_INT,
    input       TP1_INT,

    output      LVDS_0N,
    output      LVDS_0P,
    output      LVDS_1N,
    output      LVDS_1P,
    output      LVDS_2N,
    output      LVDS_2P,
    output      LVDS_3N,
    output      LVDS_3P,
    output      LVDS_4N,
    output      LVDS_4P,
    output      LVDS_5N,
    output      LVDS_5P,
    output      LVDS_CLK_N,
    output      LVDS_CLK_P,
    output      POL,
    output      TP1
);

    wire [ 5 : 0 ] lvds_in;
    wire [ 5 : 0 ] lvds_n;
    wire [ 5 : 0 ] lvds_p;

    assign lvds_in = { LVDS_5, LVDS_4, LVDS_3, LVDS_2, LVDS_1, LVDS_0 };
    assign { LVDS_5N, LVDS_4N, LVDS_3N, LVDS_2N, LVDS_1N, LVDS_0N } = lvds_n;
    assign { LVDS_5P, LVDS_4P, LVDS_3P, LVDS_2P, LVDS_1P, LVDS_0P } = lvds_p;

    genvar i_gv;

    generate
        for( i_gv = 0; i_gv < 6; i_gv = i_gv + 1 )
        begin : gen_obuftds
            OBUFTDS #(
                .IOSTANDARD ( "MINI_LVDS_25"    ))
            I_OBUFTDS_LVDS(
                .I          ( lvds_in[i_gv]     ),
                .T          ( ~OEN              ),

                .O          ( lvds_p[i_gv]      ),
                .OB         ( lvds_n[i_gv]      )
            );
        end
    endgenerate

    OBUFTDS #(
        .IOSTANDARD ( "MINI_LVDS_25"    ))
    I_OBUFTDS_LVDS_CLK(
        .I          ( LVDS_CLK          ),
        .T          ( ~OEN              ),

        .O          ( LVDS_CLK_P        ),
        .OB         ( LVDS_CLK_N        )
    );

`ifdef ZYNQ_CARRIER_CARD
    OBUFT #(
        .IOSTANDARD ( "LVCMOS25"        ),
`else
    OBUFT #(
        .IOSTANDARD ( "LVCMOS33"        ),
`endif
        .DRIVE      ( 12                ), // 4, 8, 12, or 16 mA
        .SLEW       ( "SLOW"            )) // "SLOW" or "FAST"
    I_OBUFT_POL(
        .I          ( POL_INT           ),
        .T          ( ~OEN              ),

        .O          ( POL               )
    );

`ifdef ZYNQ_CARRIER_CARD
    OBUFT #(
        .IOSTANDARD ( "LVCMOS25"        ),
`else
    OBUFT #(
        .IOSTANDARD ( "LVCMOS33"        ),
`endif
        .DRIVE      ( 12                ), // 4, 8, 12, or 16 mA
        .SLEW       ( "SLOW"            )) // "SLOW" or "FAST"
    I_OBUFT_TP1(
        .I          ( TP1_INT           ),
        .T          ( ~OEN              ),

        .O          ( TP1               )
    );

 endmodule
