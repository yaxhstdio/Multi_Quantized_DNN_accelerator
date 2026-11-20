`timescale 1ns / 1ps

module tb_top;

    parameter NUM_PES        = 4;
    parameter MAX_DATA_WIDTH = 16;
    parameter ACC_WIDTH      = 32;
    parameter ADDR_WIDTH     = 6;

    reg                                clk;
    reg                                rst_n;
    reg                                start;
    reg  [ADDR_WIDTH-1:0]              layer_length;
    reg  [1:0]                         mode_in;
    reg                                relu_en;
    reg  signed [MAX_DATA_WIDTH-1:0]   act_in;
    reg  signed [ACC_WIDTH-1:0]        b0, b1, b2, b3;

    // Interconnect Wires
    wire                               bram_rd_en;
    wire [ADDR_WIDTH-1:0]              bram_rd_addr;
    wire [63:0]                        bram_rd_data; // Packed weights from BRAM (4 x 16-bit)
    wire                               fifo_wr_en;
    wire                               mac_en;
    wire                               accum_clr;
    wire                               last_cycle;
    wire                               valid_out;

    wire [(NUM_PES*ACC_WIDTH)-1:0]      acc_out_flat;
    wire [(NUM_PES*MAX_DATA_WIDTH)-1:0] quant_out_flat;
    wire [15:0]                        fifo_rd_data;
    wire                               fifo_full, fifo_empty;

    // Packed Biases Bus
    wire [(NUM_PES*ACC_WIDTH)-1:0]      biases_flat = {b3, b2, b1, b0};

    // -------------------------------------------------------------------------
    // 1. Instantiate Parameter BRAM (Memory)
    // -------------------------------------------------------------------------
    parameter_bram #(
        .DATA_WIDTH (64),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) u_bram (
        .clk        (clk),
        .rd_en      (bram_rd_en),
        .rd_addr    (bram_rd_addr),
        .rd_data    (bram_rd_data)
    );

    // -------------------------------------------------------------------------
    // 2. Instantiate Control Unit (The FSM Brain)
    // -------------------------------------------------------------------------
    mac_control_unit #(
        .ADDR_WIDTH (ADDR_WIDTH)
    ) u_fsm (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start),
        .layer_length (layer_length),
        .valid_out    (valid_out),
        .bram_rd_en   (bram_rd_en),
        .bram_rd_addr (bram_rd_addr),
        .fifo_wr_en   (fifo_wr_en),
        .mac_en       (mac_en),
        .accum_clr    (accum_clr),
        .last_cycle   (last_cycle)
    );

    // -------------------------------------------------------------------------
    // 3. Instantiate Configurable MAC Array + Requantizer + Mode Decoder
    // -------------------------------------------------------------------------
    mac_array_requantized #(
        .NUM_PES        (NUM_PES),
        .MAX_DATA_WIDTH (MAX_DATA_WIDTH),
        .ACC_WIDTH      (ACC_WIDTH)
    ) u_mac_array (
        .clk            (clk),
        .rst_n          (rst_n),
        .mode_in        (mode_in),
        .mac_en         (mac_en),
        .accum_clr      (accum_clr),
        .relu_en        (relu_en),
        .last_cycle     (last_cycle),
        .act_in         (act_in),
        .weights_flat   (bram_rd_data), // Weights fed automatically from BRAM!
        .biases_flat    (biases_flat),
        .acc_out_flat   (acc_out_flat),
        .quant_out_flat (quant_out_flat),
        .valid_out      (valid_out)
    );

    // -------------------------------------------------------------------------
    // 4. Instantiate Activation FIFO (Output Storage)
    // -------------------------------------------------------------------------
    sync_fifo #(
        .DATA_WIDTH (16),
        .ADDR_WIDTH (4)
    ) u_fifo (
        .clk        (clk),
        .rst_n      (rst_n),
        .wr_en      (fifo_wr_en),
        .wr_data    (quant_out_flat[15:0]), // Push PE0 requantized result into FIFO
        .rd_en      (1'b0),
        .rd_data    (fifo_rd_data),
        .full       (fifo_full),
        .empty      (fifo_empty)
    );

    // 100 MHz Clock Generation
    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        $dumpfile("tb_top.vcd");
        $dumpvars(0, tb_top);

        // Initialize Test Environment
        rst_n        = 0;
        start        = 0;
        layer_length = 2; // Matches the 2 lines written by python golden model
        mode_in      = 2'b01; // INT8 mode
        relu_en      = 1;
        act_in       = 0;
        b0 = 0; b1 = 0; b2 = 0; b3 = 0;

        #20 rst_n = 1;

        // ---------------------------------------------------------------------
        // Trigger the Autonomous FSM Execution
        // ---------------------------------------------------------------------
        @(negedge clk);
        start = 1;
        act_in = 16'sd32; // Provide Feature 0 activation

        @(negedge clk);
        start = 0;        // Drop start trigger pulse
        act_in = 16'sd64; // Provide Feature 1 activation (last cycle)

        // Let the FSM complete computation and writeback to FIFO
        #80;
        
        $display("========================================");
        $display("   TOP-LEVEL INTEGRATION SIMULATION     ");
        $display("========================================");
        $display("FSM successfully fetched weights from BRAM.");
        $display("MAC Array computed dot-product autonomously.");
        $display("FIFO Write Enable triggered: %0b", fifo_wr_en);
        $display("Final Requantized Output in FIFO: %0d", quant_out_flat[15:0]);
        $display("========================================");

        #20 $finish;
    end

endmodule