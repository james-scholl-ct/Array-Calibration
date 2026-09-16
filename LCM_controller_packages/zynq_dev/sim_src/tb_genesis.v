`timescale 1ns / 1ps

`define I_PS7                   tb_genesis.DUT.genesis_i.processing_system7_0.inst
`define FPGA_RST_CTRL_DEFAULT   32'h01F33F0F
`define AXI4_GENESIS_BASE_ADDR  32'h43C0_0000
`define W2B(BASE, W)  (BASE | (W << 2))


module tb_genesis();

    // PS inputs (see UG585: 26.2.1)
    reg tb_ps_clk;
    reg tb_ps_por_b;
    reg tb_ps_srst_b;

    // AXI
    reg  [31:0] rdata;
    reg         rresp;
    reg         bresp;

    integer i;

    integer iv_file, ov_file;
    integer stat;
    reg [ 80 * 8 - 1 : 0 ] buff;
    reg [ 80 * 8 - 1 : 0 ] cmd;
    reg [ 31 : 0 ] addr;
    reg [ 31 : 0 ] data;
    reg [ 31 : 0 ] delay;

    initial
    begin
        // Assign all DUT inputs
        tb_ps_clk = 1'b0;
        tb_ps_por_b = 1'b0;
        tb_ps_srst_b = 1'b1;

        // read input vector file
        iv_file = $fopen("input_vector.txt", "r");
        if (iv_file == 0)
        begin
            $display("Couldn't open file 'input_vector.txt'\n");
            $finish;
        end

        // open output vector file
        ov_file = $fopen("output_vector.txt", "w");
        if (ov_file == 0)
        begin
            $display("Couldn't open file 'output_vector.txt'\n");
            $finish;
        end
    end

    // 33.33MHz PS clock (30ns period, 15ns edge interval)
    always #15 tb_ps_clk = ~tb_ps_clk;

    // -------------------------------------------------------------------------
    // Main
    // -------------------------------------------------------------------------
    initial
    begin
        $display("Starting simulation.");
        // Apply defaults for RST_CTRL
        `I_PS7.fpga_soft_reset(`FPGA_RST_CTRL_DEFAULT);

        // POR
        repeat(20) @(posedge tb_ps_clk);
        tb_ps_por_b = 1'b1;

        // deassert FCLK_RESET0_N (see UG585: 26.4.1 and pg 1617 in Appendix B)
        repeat(6) @(posedge tb_ps_clk);
        `I_PS7.fpga_soft_reset(`FPGA_RST_CTRL_DEFAULT & ~(1<<0));

        // wait for clk_wiz locked status
        #10000;

        stat = $fgets(buff, iv_file);

        while( !$feof(iv_file) )
        begin
            // get first word
            stat = $sscanf(buff, "%s", cmd);

            case( cmd )
                "read":
                begin
                    stat = $sscanf(buff, "%s %h", cmd, addr);
                    `I_PS7.read_data(`W2B(`AXI4_GENESIS_BASE_ADDR, addr), 4, rdata, rresp);
                    $fwrite(ov_file, "read %x %x %b\n", addr, rdata, rresp);
                    $display("%t - AXI read: addr 'd%d = 32'h%x", $time, addr, rdata);
                end

                "write":
                begin
                    stat = $sscanf(buff, "%s %h %h", cmd, addr, data);
                    `I_PS7.write_data(`W2B(`AXI4_GENESIS_BASE_ADDR, addr), 4, data, bresp);
                    $fwrite(ov_file, "write %x %x %b\n", addr, data, bresp);
                    $display("%t - AXI write: addr 'd%d = 32'h%x", $time, addr, data);
                end

                "wait":
                begin
                    stat = $sscanf(buff, "%s %d", cmd, delay);
                    $display("%t - Delay %d ns", $time, delay);
                    #(delay);
                end
            endcase

            stat = $fgets(buff, iv_file);
        end

        // runoff
        #1000;

        $fclose(iv_file);
        $fclose(ov_file);
        $display("Simulation complete.");
        $stop;
    end


    // DUT
    wire tb_ps_clk_WIRE = tb_ps_clk;
    wire tb_ps_por_b_WIRE = tb_ps_por_b;
    wire tb_ps_srst_b_WIRE = tb_ps_srst_b;

    wire [2:0] onewire_temp;
    assign (weak0, weak1) onewire_temp[0] = 1'b0;
    assign (weak0, weak1) onewire_temp[1] = 1'b1;
    assign (weak0, weak1) onewire_temp[2] = 1'b0;

    genesis_wrapper DUT (
        .DDR_addr           (),
        .DDR_ba             (),
        .DDR_cas_n          (),
        .DDR_ck_n           (),
        .DDR_ck_p           (),
        .DDR_cke            (),
        .DDR_cs_n           (),
        .DDR_dm             (),
        .DDR_dq             (),
        .DDR_dqs_n          (),
        .DDR_dqs_p          (),
        .DDR_odt            (),
        .DDR_ras_n          (),
        .DDR_reset_n        (),
        .DDR_we_n           (),
        .FIXED_IO_ddr_vrn   (),
        .FIXED_IO_ddr_vrp   (),
        .FIXED_IO_mio       (),
        .FIXED_IO_ps_clk    (tb_ps_clk_WIRE),
        .FIXED_IO_ps_porb   (tb_ps_por_b_WIRE),
        .FIXED_IO_ps_srstb  (tb_ps_srst_b_WIRE),

        .GPI_IP0            (),
        .GPI_IP1            (),
        .GPO_IP0            (),
        .GPO_IP1            ()
    );

endmodule
