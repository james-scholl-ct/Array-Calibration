`timescale 1ns / 1ps

`define I_PS7                   tb_lotus.DUT.lotus_i.processing_system7_0.inst
`define AXI4_BSC_BASE_ADDR      32'h43C0_0000
`define AXI4_SPI_BASE_ADDR      32'h43C0_1000
`define FPGA_RST_CTRL_DEFAULT   32'h01F33F0F

`define W2B(BASE, W)  (BASE | (W << 2))


module tb_lotus();

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
                "read_bsc":
                begin
                    stat = $sscanf(buff, "%s %h", cmd, addr);
                    `I_PS7.read_data(`W2B(`AXI4_BSC_BASE_ADDR, addr), 4, rdata, rresp);
                    $fwrite(ov_file, "read_bsc %x %x %b\n", addr, rdata, rresp);
                    $display("%t - AXI bsc read: addr 'd%d = 32'h%x", $time, addr, rdata);
                end

                "read_spi":
                begin
                    stat = $sscanf(buff, "%s %h", cmd, addr);
                    `I_PS7.read_data(`W2B(`AXI4_SPI_BASE_ADDR, addr), 4, rdata, rresp);
                    $fwrite(ov_file, "read_spi %x %x %b\n", addr, rdata, rresp);
                    $display("%t - AXI spi read: addr 'd%d = 32'h%x", $time, addr, rdata);
                end

                "write_bsc":
                begin
                    stat = $sscanf(buff, "%s %h %h", cmd, addr, data);
                    `I_PS7.write_data(`W2B(`AXI4_BSC_BASE_ADDR, addr), 4, data, bresp);
                    $fwrite(ov_file, "write_bsc %x %x %b\n", addr, data, bresp);
                    $display("%t - AXI bsc write: addr 'd%d = 32'h%x", $time, addr, data);
                end

                "write_spi":
                begin
                    stat = $sscanf(buff, "%s %h %h", cmd, addr, data);
                    `I_PS7.write_data(`W2B(`AXI4_SPI_BASE_ADDR, addr), 4, data, bresp);
                    $fwrite(ov_file, "write_spi %x %x %b\n", addr, data, bresp);
                    $display("%t - AXI spi write: addr 'd%d = 32'h%x", $time, addr, data);
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

    lotus_wrapper DUT (
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

        .LASER_DR1_N        (),
        .LASER_DR1_P        (),
        .LASER_DR2_N        (),
        .LASER_DR2_P        (),
        .LASER_TRIGGER      (),

        .LVDS_0N            (),
        .LVDS_0P            (),
        .LVDS_1N            (),
        .LVDS_1P            (),
        .LVDS_2N            (),
        .LVDS_2P            (),
        .LVDS_3N            (),
        .LVDS_3P            (),
        .LVDS_4N            (),
        .LVDS_4P            (),
        .LVDS_5N            (),
        .LVDS_5P            (),
        .LVDS_CLK_N         (),
        .LVDS_CLK_P         (),
        .POL                (),
        .TP1                (),

        .ONEWIRE_PULLUP_EN_B(),
        .ONEWIRE_TEMP       (onewire_temp),

        .SPI_ADC0_CHSEL     (),
        .SPI_DAISY_CLK_EN   (),
        .SPI_DAISY_EN       (),
        .SPI_MISO           (),
        .SPI_MOSI           (),
        .SPI_SCLK0          (),
        .SPI_SCLK1          (),
        .SPI_SS_B           ()
    );

endmodule
