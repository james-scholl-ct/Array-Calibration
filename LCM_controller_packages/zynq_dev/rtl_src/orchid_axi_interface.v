`timescale 1 ns / 1 ps

// AXI4-Lite:
//      * all transactions are of burst length 1
//      * all data accesses use the full width of the data bus
//      * all accesses are Non-modifiable, Non-bufferable
//      * exclusive accesses are not supported

module orchid_axi_interface #(
    // Depth of the AXI RAM in words of C_S_AXI_DATA_WIDTH bits
    parameter integer C_S_AXI_DATA_DEPTH = 72,
    // Width of AXI data bus
    parameter integer C_S_AXI_DATA_WIDTH = 32,
    // Width of AXI address bus
    parameter integer C_S_AXI_ADDR_WIDTH = 9
)
(
    // Global signals
    input   wire                                        S_AXI_ACLK,
    input   wire                                        S_AXI_ARESETN,

    // (AW) Write Address channel
    input   wire    [C_S_AXI_ADDR_WIDTH-1 : 0]          S_AXI_AWADDR,
    input   wire    [2 : 0]                             S_AXI_AWPROT,
    input   wire                                        S_AXI_AWVALID,
    output  wire                                        S_AXI_AWREADY,

    // (W) Write Data channel
    input   wire    [C_S_AXI_DATA_WIDTH-1 : 0]          S_AXI_WDATA,
    input   wire    [(C_S_AXI_DATA_WIDTH/8)-1 : 0]      S_AXI_WSTRB,
    input   wire                                        S_AXI_WVALID,
    output  wire                                        S_AXI_WREADY,

    // (B) Write Response channel
    input   wire                                        S_AXI_BREADY,
    output  wire    [1 : 0]                             S_AXI_BRESP,
    output  wire                                        S_AXI_BVALID,

    // (AR) Read Address channel
    input   wire    [C_S_AXI_ADDR_WIDTH-1 : 0]          S_AXI_ARADDR,
    input   wire    [2 : 0]                             S_AXI_ARPROT,
    input   wire                                        S_AXI_ARVALID,
    output  wire                                        S_AXI_ARREADY,

    // (R) Read Data channel
    input   wire                                        S_AXI_RREADY,
    output  wire    [C_S_AXI_DATA_WIDTH-1 : 0]          S_AXI_RDATA,
    output  wire    [1 : 0]                             S_AXI_RRESP,
    output  wire                                        S_AXI_RVALID,

    // IP connections
    output  wire    [C_S_AXI_DATA_DEPTH-1 : 0]          AXI_WRITE_ENABLES,
    output  wire    [C_S_AXI_DATA_DEPTH*C_S_AXI_DATA_WIDTH-1 : 0 ] AXI_MEMORY
);


    localparam integer BYTES_PER_WORD = C_S_AXI_DATA_WIDTH / 8;
    localparam integer WORD_ADDR_LSB = $clog2(BYTES_PER_WORD);
    localparam integer WORD_ADDR_N_BITS = C_S_AXI_ADDR_WIDTH - WORD_ADDR_LSB;


    // AXI4LITE signals
    reg     [C_S_AXI_ADDR_WIDTH-1 : 0]  axi_awaddr;
    reg                                 axi_awready;
    reg                                 axi_wready;
    reg     [1 : 0]                     axi_bresp;
    reg                                 axi_bvalid;
    reg     [C_S_AXI_ADDR_WIDTH-1 : 0]  axi_araddr;
    reg                                 axi_arready;
    reg     [C_S_AXI_DATA_WIDTH-1 : 0]  axi_rdata;
    reg     [1 : 0]                     axi_rresp;
    reg                                 axi_rvalid;

    assign S_AXI_AWREADY = axi_awready;
    assign S_AXI_WREADY  = axi_wready;
    assign S_AXI_BRESP   = axi_bresp;
    assign S_AXI_BVALID  = axi_bvalid;
    assign S_AXI_ARREADY = axi_arready;
    assign S_AXI_RDATA   = axi_rdata;
    assign S_AXI_RRESP   = axi_rresp;
    assign S_AXI_RVALID  = axi_rvalid;


    // Signals for user logic register space example
    //------------------------------------------------
    reg     [C_S_AXI_DATA_WIDTH-1:0]    slv_regs[C_S_AXI_DATA_DEPTH - 1 : 0];
    wire                                slv_reg_rden;
    wire                                slv_reg_wren;
    wire    [C_S_AXI_DATA_WIDTH-1:0]    reg_data_out;
    integer                             i_byte;
    reg                                 aw_en;

    // Implement axi_awready generation
    // axi_awready is asserted for one S_AXI_ACLK clock cycle when both
    // S_AXI_AWVALID and S_AXI_WVALID are asserted. axi_awready is
    // de-asserted when reset is low.
    always @( posedge S_AXI_ACLK )
    begin
        if ( S_AXI_ARESETN == 1'b0 )
        begin
            axi_awready <= 1'b0;
            aw_en <= 1'b1;
        end
        else
        begin
            if (~axi_awready && S_AXI_AWVALID && S_AXI_WVALID && aw_en)
            begin
                // slave is ready to accept write address when
                // there is a valid write address and write data
                // on the write address and data bus. This design
                // expects no outstanding transactions.
                axi_awready <= 1'b1;
                aw_en <= 1'b0;
            end
            else if (S_AXI_BREADY && axi_bvalid)
            begin
                aw_en <= 1'b1;
                axi_awready <= 1'b0;
            end
            else
            begin
                axi_awready <= 1'b0;
            end
        end
    end

    // Implement axi_awaddr latching
    // This process is used to latch the address when both
    // S_AXI_AWVALID and S_AXI_WVALID are valid.
    always @( posedge S_AXI_ACLK )
    begin
        if ( S_AXI_ARESETN == 1'b0 )
        begin
            axi_awaddr <= 0;
        end
        else
        begin
            if (~axi_awready && S_AXI_AWVALID && S_AXI_WVALID && aw_en)
            begin
                // Write Address latching
                axi_awaddr <= S_AXI_AWADDR;
            end
        end
    end

    // Implement axi_wready generation
    // axi_wready is asserted for one S_AXI_ACLK clock cycle when both
    // S_AXI_AWVALID and S_AXI_WVALID are asserted. axi_wready is
    // de-asserted when reset is low.
    always @( posedge S_AXI_ACLK )
    begin
        if ( S_AXI_ARESETN == 1'b0 )
        begin
            axi_wready <= 1'b0;
        end
        else
        begin
            if (~axi_wready && S_AXI_WVALID && S_AXI_AWVALID && aw_en )
            begin
                // slave is ready to accept write data when
                // there is a valid write address and write data
                // on the write address and data bus. This design
                // expects no outstanding transactions.
                axi_wready <= 1'b1;
            end
            else
            begin
                axi_wready <= 1'b0;
            end
        end
    end

    // Implement memory mapped register select and write logic generation
    // The write data is accepted and written to memory mapped registers when
    // axi_awready, S_AXI_WVALID, axi_wready and S_AXI_WVALID are asserted. Write strobes are used to
    // select byte enables of slave registers while writing.
    // These registers are cleared when reset (active low) is applied.
    // Slave register write enable is asserted when valid address and data are available
    // and the slave is ready to accept the write address and write data.
    assign slv_reg_wren = axi_wready && S_AXI_WVALID && axi_awready && S_AXI_AWVALID;

    wire [ WORD_ADDR_N_BITS - 1 : 0 ] axi_wr_word_addr;

    assign axi_wr_word_addr = axi_awaddr[C_S_AXI_ADDR_WIDTH - 1 : WORD_ADDR_LSB];

    assign AXI_WRITE_ENABLES = ( slv_reg_wren && ( |S_AXI_WSTRB ) ) << axi_wr_word_addr;

    integer i;

    always @( posedge S_AXI_ACLK )
    begin
        if ( S_AXI_ARESETN == 1'b0 )
        begin
            for( i = 0; i < C_S_AXI_DATA_DEPTH; i = i + 1 )
            begin
                slv_regs[i] <= 'd0;
            end
        end
        else
        begin
            for( i = 0; i < C_S_AXI_DATA_DEPTH; i = i + 1 )
            begin
                if( AXI_WRITE_ENABLES[i] )
                begin
                    for( i_byte = 0; i_byte <= (C_S_AXI_DATA_WIDTH/8)-1; i_byte = i_byte+1 )
                    begin
                        if( S_AXI_WSTRB[i_byte] == 1'b1 )
                        begin
                            slv_regs[i][(i_byte*8) +: 8] <= S_AXI_WDATA[(i_byte*8) +: 8];
                        end
                    end
                end
            end
        end
    end

    // Implement write response logic generation
    // The write response and response valid signals are asserted by the slave
    // when axi_wready, S_AXI_WVALID, axi_wready and S_AXI_WVALID are asserted.
    // This marks the acceptance of address and indicates the status of
    // write transaction.
    always @( posedge S_AXI_ACLK )
    begin
        if ( S_AXI_ARESETN == 1'b0 )
        begin
            axi_bvalid  <= 0;
            axi_bresp   <= 2'b0;
        end
        else
        begin
            if (axi_awready && S_AXI_AWVALID && ~axi_bvalid && axi_wready && S_AXI_WVALID)
            begin
                // indicates a valid write response is available
                axi_bvalid <= 1'b1;
                axi_bresp  <= 2'b0; // 'OKAY' response
            end
            else if (S_AXI_BREADY && axi_bvalid)
            begin
                //check if bready is asserted while bvalid is high)
                //(there is a possibility that bready is always asserted high)
                axi_bvalid <= 1'b0;
            end
        end
    end

    // Implement axi_arready generation
    // axi_arready is asserted for one S_AXI_ACLK clock cycle when
    // S_AXI_ARVALID is asserted. axi_awready is
    // de-asserted when reset (active low) is asserted.
    // The read address is also latched when S_AXI_ARVALID is
    // asserted. axi_araddr is reset to zero on reset assertion.
    always @( posedge S_AXI_ACLK )
    begin
        if ( S_AXI_ARESETN == 1'b0 )
        begin
            axi_arready <= 1'b0;
            axi_araddr  <= 32'b0;
        end
        else
        begin
            if (~axi_arready && S_AXI_ARVALID)
            begin
                // indicates that the slave has acceped the valid read address
                axi_arready <= 1'b1;
                // Read address latching
                axi_araddr  <= S_AXI_ARADDR;
            end
            else
            begin
                axi_arready <= 1'b0;
            end
        end
    end

    // Implement axi_arvalid generation
    // axi_rvalid is asserted for one S_AXI_ACLK clock cycle when both
    // S_AXI_ARVALID and axi_arready are asserted. The slave registers
    // data are available on the axi_rdata bus at this instance. The
    // assertion of axi_rvalid marks the validity of read data on the
    // bus and axi_rresp indicates the status of read transaction.axi_rvalid
    // is deasserted on reset (active low). axi_rresp and axi_rdata are
    // cleared to zero on reset (active low).
    always @( posedge S_AXI_ACLK )
    begin
        if ( S_AXI_ARESETN == 1'b0 )
        begin
            axi_rvalid <= 0;
            axi_rresp  <= 0;
        end
        else
        begin
            if (axi_arready && S_AXI_ARVALID && ~axi_rvalid)
            begin
                // Valid read data is available at the read data bus
                axi_rvalid <= 1'b1;
                axi_rresp  <= 2'b0; // 'OKAY' response
            end
            else if (axi_rvalid && S_AXI_RREADY)
            begin
                // Read data is accepted by the master
                axi_rvalid <= 1'b0;
            end
        end
    end

    // Implement memory mapped register select and read logic generation
    // Slave register read enable is asserted when valid address is available
    // and the slave is ready to accept the read address.
    assign slv_reg_rden = axi_arready & S_AXI_ARVALID & ~axi_rvalid;

    wire [ WORD_ADDR_N_BITS - 1 : 0 ] axi_rd_word_addr;

    assign axi_rd_word_addr = axi_araddr[C_S_AXI_ADDR_WIDTH - 1 : WORD_ADDR_LSB];

    wire [ C_S_AXI_DATA_WIDTH - 1 : 0 ] sum;

    assign sum = slv_regs[64] + slv_regs[65];

    assign reg_data_out = ( axi_rd_word_addr == 'd66 ) ? sum : slv_regs[axi_rd_word_addr];

    // Output register or memory read data
    always @( posedge S_AXI_ACLK )
    begin
        if ( S_AXI_ARESETN == 1'b0 )
        begin
            axi_rdata <= 0;
        end
        else
        begin
            // When there is a valid read address (S_AXI_ARVALID) with
            // acceptance of read address by the slave (axi_arready),
            // output the read dada
            if (slv_reg_rden)
            begin
                axi_rdata <= reg_data_out;     // register read data
            end
        end
    end

    // Add user logic here
    genvar i_gv;
    generate
        for( i_gv = 0; i_gv < C_S_AXI_DATA_DEPTH; i_gv = i_gv + 1 )
        begin : pack
            assign AXI_MEMORY[ C_S_AXI_DATA_WIDTH * (i_gv+1) - 1 :
                               C_S_AXI_DATA_WIDTH * i_gv         ] = slv_regs[ i_gv ];
        end
    endgenerate

endmodule
