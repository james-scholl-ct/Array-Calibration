`include "delorean_defines.v"
`timescale 1 ns / 1 ps

module delorean_dma_bram #(
    parameter integer WIDTH = 32
)
(
    input   wire    [9 : 0]                             A_ADDR,
    input   wire                                        A_CLK,
    input   wire    [WIDTH - 1 : 0]                     A_DI,
    input   wire                                        A_EN,
    input   wire    [WIDTH / 8 - 1 : 0]                 A_WE,
    output  reg     [WIDTH - 1 : 0]                     A_DO,

    input   wire    [9 : 0]                             B_ADDR,
    input   wire                                        B_CLK,
    input   wire    [WIDTH - 1 : 0]                     B_DI,
    input   wire                                        B_EN,
    input   wire    [WIDTH / 8 - 1 : 0]                 B_WE,
    output  reg     [WIDTH - 1 : 0]                     B_DO
);


    // NO_CHANGE write mode: maintains the output from previous read.
    // Don't implement byte write enables.
    reg [WIDTH - 1 : 0] ram [1023 : 0];
    wire a_we_any;
    wire b_we_any;

    assign a_we_any = |A_WE;
    assign b_we_any = |B_WE;

    always @(posedge A_CLK)
    begin
        if (A_EN)
        begin
            if (a_we_any)
            begin
                ram[A_ADDR] <= A_DI;
            end
            else
            begin
                A_DO <= ram[A_ADDR];
            end
        end
    end

    always @(posedge B_CLK)
    begin
        if (B_EN)
        begin
            if (b_we_any)
            begin
                ram[B_ADDR] <= B_DI;
            end
            else
            begin
                B_DO <= ram[B_ADDR];
            end
        end
    end

endmodule
