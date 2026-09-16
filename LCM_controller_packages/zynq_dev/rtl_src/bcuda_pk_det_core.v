`include "bcuda_defines.v"
`timescale 1 ns / 1 ps

module bcuda_pk_det_core #(
    // Data path values
    parameter integer DIN_N_BITS = 16,
    parameter integer THRESH_N_BITS = 16,
    parameter integer PEAK_VAL_N_BITS = 16,
    // Index up to 400 samples * 8 interp = 3200 samples
    parameter integer GUARD_SAMPLES_N_BITS = 12,
    parameter integer SAMPLE_COUNT_N_BITS = 12,
    parameter integer PEAK_IDX_N_BITS = 12,
    //
    parameter integer NUM_WINDOWS_N_BITS = 8,
    parameter integer NOISE_EST_N_BITS = 16,
    parameter integer BUNDLE_N_BITS = 88
)
(
    input   wire                                            ACK,
    input   wire                                            CLK,
    input   wire signed [DIN_N_BITS - 1 : 0]                DIN,
    input   wire                                            DIN_DV,
    input   wire        [GUARD_SAMPLES_N_BITS - 1 : 0]      GUARD_SAMPLES,   // reduce by 1 (10 -> 9)
    input   wire                                            MODE,
    input   wire        [SAMPLE_COUNT_N_BITS - 1 : 0]       SAMPLE_IDX_OFF,
    input   wire        [SAMPLE_COUNT_N_BITS - 1 : 0]       SAMPLE_IDX_ON,
    input   wire        [NUM_WINDOWS_N_BITS - 1 : 0]        NUM_WINDOWS,     // TODO: implement CFAR
    input   wire signed [THRESH_N_BITS - 1 : 0]             PROG_FA_THRESH,
    input   wire                                            PROG_FA_THRESH_EN,
    input   wire                                            RESET,

    output  wire signed [THRESH_N_BITS - 1 : 0]             ADAPTIVE_THRESH, // TODO: implement CFAR
    output  wire        [BUNDLE_N_BITS - 1 : 0]             BUNDLE,
    output  wire signed [NOISE_EST_N_BITS - 1 : 0]          NOISE_EST,       // TODO: implement CFAR
    output  reg                                             PEAK_DET_A,
    output  reg         [PEAK_IDX_N_BITS - 1 : 0]           PEAK_IDX_A,
    output  reg  signed [PEAK_VAL_N_BITS - 1 : 0]           PEAK_VAL_A,
    output  reg                                             PEAK_DET_B,
    output  reg         [PEAK_IDX_N_BITS - 1 : 0]           PEAK_IDX_B,
    output  reg  signed [PEAK_VAL_N_BITS - 1 : 0]           PEAK_VAL_B,
    output  reg                                             REQ
);


    // In the mode for the two greatest returns, each detector requires that
    //  1) the current sample is greater than a threshold
    //  2) the current sample is greater than the target detector's current max
    //
    // In the mode for greatest and latest returns, amend requirement (2) above
    // such that the target detector always updates with the latest sample and
    // the other detector holds the maximum up to, but excluding, the latest
    // sample. For the latest return, compare only against the current local
    // maximum and not against the previous saved maximum in order to capture
    // the top of the current peak, which may be less than a previous peak.
    //
    // A detector is enabled over the span of a local maximum. Once the end of
    // a local maximum is detected, according to guard_count, the selection of
    // the target detector is updated in preparation for the next peak.
    wire                                        mode_greatest;
    wire signed [THRESH_N_BITS - 1 : 0]         thresh_eff;
    wire                                        din_over_thresh;
    wire                                        detector_a_max_test;
    wire                                        detector_b_max_test;
    wire                                        range_gate_done;
    wire                                        det_a_en;
    wire                                        det_b_en;
    reg         [SAMPLE_COUNT_N_BITS - 1 : 0]   sample_count;
    reg         [SAMPLE_COUNT_N_BITS - 1 : 0]   guard_count;
    reg                                         use_b_detector;
    reg  signed [PEAK_VAL_N_BITS - 1 : 0]       local_max;

    assign mode_greatest = (MODE == 'd0);
    assign thresh_eff = PROG_FA_THRESH_EN ? PROG_FA_THRESH : ADAPTIVE_THRESH;
    assign din_over_thresh = (DIN > thresh_eff);
    assign detector_a_max_test = (mode_greatest) ? (DIN > PEAK_VAL_A)
                                                 : (DIN > local_max);
    assign detector_b_max_test = (mode_greatest) ? (DIN > PEAK_VAL_B)
                                                 : (DIN > local_max);
    assign range_gate_done = (sample_count >= SAMPLE_IDX_ON);

    assign det_a_en = din_over_thresh &&
                      detector_a_max_test &&
                      ~use_b_detector &&
                      range_gate_done;

    assign det_b_en = din_over_thresh &&
                      detector_b_max_test &&
                      use_b_detector &&
                      range_gate_done;

    always @(posedge CLK)
    begin
        if (RESET || ACK && REQ)
        begin
            sample_count    <= 'd0;
            guard_count     <= 'd0;
            use_b_detector  <= 1'b0;
            local_max       <= $signed({1'b1, {(PEAK_VAL_N_BITS - 1){1'b0}}});
            REQ             <= 'd0;
            PEAK_DET_A      <= 1'b0;
            PEAK_IDX_A      <= 'd0;
            PEAK_VAL_A      <= $signed({1'b1, {(PEAK_VAL_N_BITS - 1){1'b0}}});
            PEAK_DET_B      <= 1'b0;
            PEAK_IDX_B      <= 'd0;
            PEAK_VAL_B      <= $signed({1'b1, {(PEAK_VAL_N_BITS - 1){1'b0}}});
        end
        else if (DIN_DV && !REQ)
        begin
            sample_count <= sample_count + 'd1;

            REQ <= (sample_count == SAMPLE_IDX_OFF);

            // Kick (or restart) counter on each update. When counter expires,
            // switch detectors. GUARD_SAMPLES must be >= 1 (i.e. 2 samples).
            if (det_a_en || det_b_en)
            begin
                guard_count <= 'd1;
                local_max   <= DIN;
            end
            else if (guard_count == GUARD_SAMPLES)
            begin
                guard_count <= 'd0;
                local_max   <= $signed({1'b1, {(PEAK_VAL_N_BITS - 1){1'b0}}});
                use_b_detector <= (PEAK_VAL_B < PEAK_VAL_A);
            end
            else if (guard_count != 'd0)
            begin
                guard_count <= guard_count + 'd1;
            end

            PEAK_DET_A <= det_a_en;
            if (det_a_en)
            begin
                PEAK_IDX_A  <= sample_count;
                PEAK_VAL_A  <= DIN;
            end

            PEAK_DET_B <= det_b_en;
            if (det_b_en)
            begin
                PEAK_IDX_B  <= sample_count;
                PEAK_VAL_B  <= DIN;
            end
        end
    end

    // TODO: implement CFAR
    assign ADAPTIVE_THRESH = 0;

    assign NOISE_EST = 0;

    assign BUNDLE = {'d0,
                     ADAPTIVE_THRESH,
                     NOISE_EST,
                     PEAK_IDX_A,
                     PEAK_VAL_A,
                     PEAK_IDX_B,
                     PEAK_VAL_B};

endmodule
