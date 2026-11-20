`timescale 1ns / 1ps

module tb_phase1;

    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH  = 24;
    parameter SHIFT_BITS = 7; // Divide by 128

    reg                         clk;
    reg                         rst_n;
    reg                         mac_en;
    reg                         accum_clr;
    reg  signed [DATA_WIDTH-1:0] act_in;
    reg  signed [DATA_WIDTH-1:0] weight_in;
    reg  signed [ACC_WIDTH-1:0]  bias_in;
    reg                         relu_en;

    wire signed [ACC_WIDTH-1:0]  accum_out;
    wire signed [DATA_WIDTH-1:0] quant_out;

    // Instantiate MAC Unit
    serial_mac #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH)
    ) u_mac (
        .clk       (clk),
        .rst_n     (rst_n),
        .mac_en    (mac_en),
        .accum_clr (accum_clr),
        .act_in    (act_in),
        .weight_in (weight_in),
        .bias_in   (bias_in),
        .accum_out (accum_out)
    );

    // Instantiate Requantize Block
    requantize #(
        .ACC_WIDTH(ACC_WIDTH),
        .OUT_WIDTH(DATA_WIDTH),
        .SHIFT_BITS(SHIFT_BITS)
    ) u_req (
        .relu_en   (relu_en),
        .accum_in  (accum_out),
        .quant_out (quant_out)
    );

    // 100 MHz Clock Generation
    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        $dumpfile("phase1_mac_req.vcd");
        $dumpvars(0, tb_phase1);

        // Initialize signals
        rst_n     = 0;
        mac_en    = 0;
        accum_clr = 0;
        act_in    = 0;
        weight_in = 0;
        bias_in   = 0;
        relu_en   = 1;

        #20 rst_n = 1;

        // --------------------------------------------------------------------
        // Test 1: 4-Element Dot Product (Normal Range)
        // Bias = 256
        // Cycle 1: 20 * 16  = 320   (Accum = 576)
        // Cycle 2: 40 * 32  = 1280  (Accum = 1856)
        // Cycle 3: -10 * 16 = -160  (Accum = 1696)
        // Cycle 4: 50 * 8   = 400   (Accum = 2096)
        // Expected Requantized: (2096 + 64) >>> 7 = 2160 / 128 = 16
        // --------------------------------------------------------------------
        @(negedge clk);
        mac_en = 1; accum_clr = 1; bias_in = 24'sd256;
        act_in = 8'sd20; weight_in = 8'sd16;

        @(negedge clk);
        accum_clr = 0;
        act_in = 8'sd40; weight_in = 8'sd32;

        @(negedge clk);
        act_in = -8'sd10; weight_in = 8'sd16;

        @(negedge clk);
        act_in = 8'sd50; weight_in = 8'sd8;

        @(negedge clk);
        mac_en = 0;
        #1;
        $display("[Test 1] Normal Dot Product | Accum: %0d (Exp: 2096) | Quant: %0d (Exp: 16)", accum_out, quant_out);

        // --------------------------------------------------------------------
        // Test 2: Positive Saturation Overflow
        // Cycle 1: 120 * 100 = 12000 (with Bias = 0)
        // Cycle 2: 120 * 100 = 12000 (Accum = 24000)
        // Shifted: (24000 + 64) >>> 7 = 188 -> Should saturate to +127
        // --------------------------------------------------------------------
        @(negedge clk);
        mac_en = 1; accum_clr = 1; bias_in = 24'sd0;
        act_in = 8'sd120; weight_in = 8'sd100;

        @(negedge clk);
        accum_clr = 0;
        act_in = 8'sd120; weight_in = 8'sd100;

        @(negedge clk);
        mac_en = 0;
        #1;
        $display("[Test 2] Pos Saturation     | Accum: %0d (Exp: 24000)| Quant: %0d (Exp: 127)", accum_out, quant_out);

        // --------------------------------------------------------------------
        // Test 3: Negative Result with ReLU Enabled vs. Disabled
        // Cycle 1: 50 * (-40) = -2000
        // Shifted: (-2000 + 64) >>> 7 = -16
        // --------------------------------------------------------------------
        @(negedge clk);
        mac_en = 1; accum_clr = 1; bias_in = 24'sd0;
        act_in = 8'sd50; weight_in = -8'sd40;

        @(negedge clk);
        mac_en = 0;
        #1;
        $display("[Test 3a] Negative (ReLU=1) | Accum: %0d (Exp: -2000)| Quant: %0d (Exp: 0)", accum_out, quant_out);

        relu_en = 0;
        #5;
        $display("[Test 3b] Negative (ReLU=0) | Accum: %0d (Exp: -2000)| Quant: %0d (Exp: -16)", accum_out, quant_out);

        #20 $finish;
    end

endmodule