`include "bcuda_defines.v"
`timescale 1 ns / 1 ps

// A wrapper for eight instances of the matched filter.

module bcuda_pk_det #(
    parameter integer DIN_N_BITS = 16,
    parameter integer THRESH_N_BITS = 16,
    parameter integer PEAK_VAL_N_BITS = 16,
    parameter integer GUARD_SAMPLES_N_BITS = 12,
    parameter integer SAMPLE_COUNT_N_BITS = 12,
    parameter integer PEAK_IDX_N_BITS = 12,
    parameter integer NUM_WINDOWS_N_BITS = 8,
    parameter integer NOISE_EST_N_BITS = 16,
    parameter integer BUNDLE_N_BITS = 88,
    parameter integer DP_ADDR_N_BITS = 4,
    parameter integer DP_DIN_N_BITS = 128
)
(
    input   wire                                            CLK,
    input   wire                                            RESET,
    input   wire signed [DIN_N_BITS - 1 : 0]                DIN0,
    input   wire                                            DIN0_DV,
    input   wire signed [DIN_N_BITS - 1 : 0]                DIN1,
    input   wire                                            DIN1_DV,
    input   wire signed [DIN_N_BITS - 1 : 0]                DIN2,
    input   wire                                            DIN2_DV,
    input   wire signed [DIN_N_BITS - 1 : 0]                DIN3,
    input   wire                                            DIN3_DV,
    input   wire signed [DIN_N_BITS - 1 : 0]                DIN4,
    input   wire                                            DIN4_DV,
    input   wire signed [DIN_N_BITS - 1 : 0]                DIN5,
    input   wire                                            DIN5_DV,
    input   wire signed [DIN_N_BITS - 1 : 0]                DIN6,
    input   wire                                            DIN6_DV,
    input   wire signed [DIN_N_BITS - 1 : 0]                DIN7,
    input   wire                                            DIN7_DV,
    input   wire        [GUARD_SAMPLES_N_BITS - 1 : 0]      GUARD_SAMPLES,
    input   wire                                            MODE,
    input   wire        [SAMPLE_COUNT_N_BITS - 1 : 0]       SAMPLE_IDX_OFF,
    input   wire        [SAMPLE_COUNT_N_BITS - 1 : 0]       SAMPLE_IDX_ON,
    input   wire        [NUM_WINDOWS_N_BITS - 1 : 0]        NUM_WINDOWS,
    input   wire signed [THRESH_N_BITS - 1 : 0]             PROG_FA_THRESH,
    input   wire                                            PROG_FA_THRESH_EN,

    // PBE RAM interface
    output  reg                                             BLOCK_VALID,
    output  reg         [DP_ADDR_N_BITS - 1 : 0]            DP_ADDR,
    output  wire        [DP_DIN_N_BITS - 1 : 0]             DP_DIN,
    output  wire                                            DP_WE
);


    genvar i_gv;
    localparam integer N_CHANNELS = 8;

    wire signed [DIN_N_BITS - 1 : 0]    din_vec [N_CHANNELS - 1 : 0];
    wire        [N_CHANNELS - 1 : 0]    din_dv_vec;
    wire        [BUNDLE_N_BITS - 1 : 0] bundle_vec [N_CHANNELS - 1 : 0];
    wire        [N_CHANNELS - 1 : 0]    ack_vec;
    wire        [N_CHANNELS - 1 : 0]    req_vec;

    assign din_vec[0] = DIN0;
    assign din_vec[1] = DIN1;
    assign din_vec[2] = DIN2;
    assign din_vec[3] = DIN3;
    assign din_vec[4] = DIN4;
    assign din_vec[5] = DIN5;
    assign din_vec[6] = DIN6;
    assign din_vec[7] = DIN7;
    assign din_dv_vec[0] = DIN0_DV;
    assign din_dv_vec[1] = DIN1_DV;
    assign din_dv_vec[2] = DIN2_DV;
    assign din_dv_vec[3] = DIN3_DV;
    assign din_dv_vec[4] = DIN4_DV;
    assign din_dv_vec[5] = DIN5_DV;
    assign din_dv_vec[6] = DIN6_DV;
    assign din_dv_vec[7] = DIN7_DV;

    generate
        for (i_gv = 0; i_gv < N_CHANNELS; i_gv = i_gv + 1)
        begin : gen_pk_det_core
            bcuda_pk_det_core #(
                .DIN_N_BITS             (DIN_N_BITS),
                .THRESH_N_BITS          (THRESH_N_BITS),
                .PEAK_VAL_N_BITS        (PEAK_VAL_N_BITS),
                .GUARD_SAMPLES_N_BITS   (GUARD_SAMPLES_N_BITS),
                .SAMPLE_COUNT_N_BITS    (SAMPLE_COUNT_N_BITS),
                .PEAK_IDX_N_BITS        (PEAK_IDX_N_BITS),
                .NUM_WINDOWS_N_BITS     (NUM_WINDOWS_N_BITS),
                .NOISE_EST_N_BITS       (NOISE_EST_N_BITS),
                .BUNDLE_N_BITS          (BUNDLE_N_BITS))
            i_bcuda_pk_det_core (
                .ACK                    (ack_vec[i_gv]),
                .CLK                    (CLK),
                .DIN                    (din_vec[i_gv]),
                .DIN_DV                 (din_dv_vec[i_gv]),
                .GUARD_SAMPLES          (GUARD_SAMPLES),
                .MODE                   (MODE),
                .SAMPLE_IDX_OFF         (SAMPLE_IDX_OFF),
                .SAMPLE_IDX_ON          (SAMPLE_IDX_ON),
                .NUM_WINDOWS            (NUM_WINDOWS),
                .PROG_FA_THRESH         (PROG_FA_THRESH),
                .PROG_FA_THRESH_EN      (PROG_FA_THRESH_EN),
                .RESET                  (RESET),

                .ADAPTIVE_THRESH        (),
                .BUNDLE                 (bundle_vec[i_gv]),
                .NOISE_EST              (),
                .PEAK_DET_A             (),
                .PEAK_IDX_A             (),
                .PEAK_VAL_A             (),
                .PEAK_DET_B             (),
                .PEAK_IDX_B             (),
                .PEAK_VAL_B             (),
                .REQ                    (req_vec[i_gv])
            );
        end
    endgenerate


    // ------------------------------------------------------------------------
    // Interleave (lidar return) bundles into SDP RAM
    //     * Two partitions of 2**(DP_ADDR_N_BITS - 1) words each
    //     * DP_DIN_N_BITS per word
    // ------------------------------------------------------------------------
    wire [DP_ADDR_N_BITS - 2 : 0]   channel_sel;
    wire                            req_active;

    assign channel_sel = DP_ADDR[DP_ADDR_N_BITS - 2 : 0];

    assign req_active = req_vec[channel_sel];

    assign ack_vec = 1'b1 << channel_sel;

    always @(posedge CLK)
    begin
        if (RESET)
        begin
            DP_ADDR <= 'd0;
            BLOCK_VALID <= 1'b0;
        end
        else
        begin
            if (req_active)
            begin
                // add modulo the ram depth
                DP_ADDR <= DP_ADDR + 'd1;
            end

            BLOCK_VALID <= req_active && (&channel_sel);
        end
    end

    // Zero-extend bundle to DP_DIN_N_BITS
    assign DP_DIN = bundle_vec[channel_sel];

    assign DP_WE = req_active;

endmodule
