`timescale 1ns / 1ps

module tb_phase2;

    parameter NUM_PES        = 4;
    parameter MAX_DATA_WIDTH = 16;
    parameter ACC_WIDTH      = 32;

    reg                                clk;
    reg                                rst_n;
    reg  [1:0]                         mode_in;
    reg                                mac_en;
    reg                                accum_clr;
    reg                                relu_en;
    reg                                last_cycle;
    reg  signed [MAX_DATA_WIDTH-1:0]   act_in;

    // Individual PE weights & biases for easy testbench driving
    reg  signed [MAX_DATA_WIDTH-1:0]   w0, w1, w2, w3;
    reg  signed [ACC_WIDTH-1:0]        b0, b1, b2, b3;

    wire [(NUM_PES*MAX_DATA_WIDTH)-1:0] weights_flat = {w3, w2, w1, w0};
    wire [(NUM_PES*ACC_WIDTH)-1:0]      biases_flat  = {b3, b2, b1, b0};

    wire [(NUM_PES*ACC_WIDTH)-1:0]      acc_out_flat;
    wire [(NUM_PES*MAX_DATA_WIDTH)-1:0] quant_out_flat;
    wire                                valid_out;

    // Unpack outputs for easy viewing in GTKWave
    wire signed [ACC_WIDTH-1:0]      pe0_acc   = acc_out_flat[31:0];
    wire signed [ACC_WIDTH-1:0]      pe1_acc   = acc_out_flat[63:32];
    wire signed [ACC_WIDTH-1:0]      pe2_acc   = acc_out_flat[95:64];
    wire signed [ACC_WIDTH-1:0]      pe3_acc   = acc_out_flat[127:96];

    wire signed [MAX_DATA_WIDTH-1:0] pe0_quant = quant_out_flat[15:0];
    wire signed [MAX_DATA_WIDTH-1:0] pe1_quant = quant_out_flat[31:16];
    wire signed [MAX_DATA_WIDTH-1:0] pe2_quant = quant_out_flat[47:32];
    wire signed [MAX_DATA_WIDTH-1:0] pe3_quant = quant_out_flat[63:48];

    // Instantiate Top-Level MAC Array
    mac_array_requantized #(
        .NUM_PES        (NUM_PES),
        .MAX_DATA_WIDTH (MAX_DATA_WIDTH),
        .ACC_WIDTH      (ACC_WIDTH)
    ) dut (
        .clk            (clk),
        .rst_n          (rst_n),
        .mode_in        (mode_in),
        .mac_en         (mac_en),
        .accum_clr      (accum_clr),
        .relu_en        (relu_en),
        .last_cycle     (last_cycle),
        .act_in         (act_in),
        .weights_flat   (weights_flat),
        .biases_flat    (biases_flat),
        .acc_out_flat   (acc_out_flat),
        .quant_out_flat (quant_out_flat),
        .valid_out      (valid_out)
    );

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        $dumpfile("phase2_array.vcd");
        $dumpvars(0, tb_phase2);

        rst_n = 0; mode_in = 2'b01; mac_en = 0; accum_clr = 0;
        relu_en = 1; last_cycle = 0; act_in = 0;
        w0 = 0; w1 = 0; w2 = 0; w3 = 0;
        b0 = 0; b1 = 0; b2 = 0; b3 = 0;

        #20 rst_n = 1;

        // ====================================================================
        // TEST 1: INT8 Mode (mode_in = 2'b01, Shift = 7 -> Divide by 128)
        // Broadcast act_in = 32 (Cycle 1), 64 (Cycle 2) across 4 different neurons
        // ====================================================================
        @(negedge clk);
        mode_in = 2'b01; mac_en = 1; accum_clr = 1; last_cycle = 0;
        act_in  = 16'sd32;
        w0 = 16'sd16;  w1 = 16'sd32;  w2 = -16'sd16; w3 = 16'sd100;

        @(negedge clk);
        accum_clr = 0; last_cycle = 1;
        act_in    = 16'sd64;
        w0 = 16'sd16;  w1 = 16'sd32;  w2 = -16'sd16; w3 = 16'sd100;

        @(negedge clk);
        mac_en = 0; last_cycle = 0;
        #1;
        $display("=== TEST 1: INT8 MODE (valid_out=%0b) ===", valid_out);
        $display("PE0 (Normal) : Acc=%0d (Exp: 1536)  | Quant=%0d (Exp: 12)",  pe0_acc, pe0_quant);
        $display("PE1 (Normal) : Acc=%0d (Exp: 3072)  | Quant=%0d (Exp: 24)",  pe1_acc, pe1_quant);
        $display("PE2 (ReLU 0) : Acc=%0d (Exp: -1536) | Quant=%0d (Exp: 0)",   pe2_acc, pe2_quant);
        $display("PE3 (PosSat) : Acc=%0d (Exp: 9600)  | Quant=%0d (Exp: 75)",  pe3_acc, pe3_quant);

        // ====================================================================
        // TEST 2: INT4 Mode (mode_in = 2'b00, Shift = 3 -> Divide by 8, Max = +7)
        // ====================================================================
        @(negedge clk);
        mode_in = 2'b00; mac_en = 1; accum_clr = 1; last_cycle = 0;
        act_in  = 16'sd4;
        w0 = 16'sd2;   w1 = 16'sd7;   w2 = -16'sd4;  w3 = 16'sd6;

        @(negedge clk);
        accum_clr = 0; last_cycle = 1;
        act_in    = 16'sd4;
        w0 = 16'sd2;   w1 = 16'sd7;   w2 = -16'sd4;  w3 = 16'sd6;

        @(negedge clk);
        mac_en = 0; last_cycle = 0;
        #1;
        $display("=== TEST 2: INT4 MODE (valid_out=%0b) ===", valid_out);
        $display("PE0 (Normal) : Acc=%0d (Exp: 16)  | Quant=%0d (Exp: 2)", pe0_acc, pe0_quant);
        $display("PE1 (INT4Sat): Acc=%0d (Exp: 56)  | Quant=%0d (Exp: 7)", pe1_acc, pe1_quant);
        $display("PE2 (ReLU 0) : Acc=%0d (Exp: -32) | Quant=%0d (Exp: 0)", pe2_acc, pe2_quant);
        $display("PE3 (Normal) : Acc=%0d (Exp: 48)  | Quant=%0d (Exp: 6)", pe3_acc, pe3_quant);

        #20 $finish;
    end

endmodule