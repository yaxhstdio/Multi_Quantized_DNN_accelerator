`timescale 1ns / 1ps

module tb_hil_digits;

    localparam NUM_IMAGES = 360;
    localparam ADDR_WIDTH = 9;

    reg                  clk;
    reg                  rst_n;
    reg                  start;
    reg [ADDR_WIDTH-1:0] base_addr;
    reg [ADDR_WIDTH-1:0] layer_length;
    reg [1:0]            mode_in;
    reg [4:0]            shift_in;
    reg                  relu_en;
    reg signed [15:0]    act_in;
    reg                  fifo_rd_en;

    wire [63:0]          fifo_rd_data;
    wire                 fifo_full;
    wire                 fifo_empty;

    reg signed [15:0]    test_inputs      [0:(NUM_IMAGES*64)-1];
    reg signed [15:0]    expected_outputs [0:(NUM_IMAGES*10)-1];
    reg [3:0]            test_labels      [0:NUM_IMAGES-1];

    reg signed [15:0]    hidden_act       [0:15];
    reg signed [15:0]    rtl_logits       [0:11];

    // Runtime configuration via plusargs
    integer cfg_mode_l1, cfg_shift_l1;
    integer cfg_mode_l2, cfg_shift_l2;

    // Cycle counter
    integer cycle_count;
    integer img_start_cycle, total_inf_cycles;

    accelerator_top #(
        .NUM_PES        (4),
        .MAX_DATA_WIDTH (16),
        .ACC_WIDTH      (48),
        .ADDR_WIDTH     (ADDR_WIDTH)
    ) u_dut (
        .clk          (clk),
        .rst_n        (rst_n),
        .start        (start),
        .base_addr    (base_addr),
        .layer_length (layer_length),
        .mode_in      (mode_in),
        .shift_in     (shift_in),
        .relu_en      (relu_en),
        .act_in       (act_in),
        .fifo_rd_en   (fifo_rd_en),
        .fifo_rd_data (fifo_rd_data),
        .fifo_full    (fifo_full),
        .fifo_empty   (fifo_empty)
    );

    initial clk = 1'b0;
    always #5 clk = ~clk;

    always @(posedge clk) begin
        if (!rst_n) cycle_count <= 0;
        else        cycle_count <= cycle_count + 1;
    end

    task run_pass;
        input [ADDR_WIDTH-1:0] t_base;
        input [ADDR_WIDTH-1:0] t_len;
        input [1:0]            t_mode;
        input [4:0]            t_shift;
        input                  t_relu;
        input                  is_layer2;
        input integer          img_idx;
        output [63:0]          pass_out;

        integer k;
        begin
            @(negedge clk);
            base_addr    = t_base;
            layer_length = t_len;
            mode_in      = t_mode;
            shift_in     = t_shift;
            relu_en      = t_relu;
            start        = 1'b1;

            @(negedge clk);
            start = 1'b0;

            k = 0;
            while (k < t_len) begin
                if (u_dut.mac_en) begin
                    act_in = is_layer2 ? hidden_act[k] : test_inputs[img_idx*64 + k];
                    k = k + 1;
                end
                @(negedge clk);
            end
            act_in = 16'sd0;

            while (fifo_empty) @(negedge clk);

            fifo_rd_en = 1'b1;
            @(negedge clk);
            pass_out   = fifo_rd_data;
            fifo_rd_en = 1'b0;
            @(negedge clk);
        end
    endtask

    integer img, p, n;
    integer total_outputs, bit_exact_matches;
    integer correct_preds, pred_class;
    reg signed [15:0] max_logit;
    reg [63:0]        fifo_word;

    initial begin
        // Defaults (INT8) if no plusargs given
        cfg_mode_l1  = 1; cfg_shift_l1 = 11;
        cfg_mode_l2  = 1; cfg_shift_l2 = 6;
        if (!$value$plusargs("MODE_L1=%d",  cfg_mode_l1))  ;
        if (!$value$plusargs("SHIFT_L1=%d", cfg_shift_l1)) ;
        if (!$value$plusargs("MODE_L2=%d",  cfg_mode_l2))  ;
        if (!$value$plusargs("SHIFT_L2=%d", cfg_shift_l2)) ;

        $readmemh("test_inputs.hex",      test_inputs);
        $readmemh("expected_outputs.hex", expected_outputs);
        $readmemh("test_labels.hex",      test_labels);

        rst_n             = 1'b0;
        start             = 1'b0;
        base_addr         = 0;
        layer_length      = 0;
        mode_in           = cfg_mode_l1[1:0];
        shift_in          = cfg_shift_l1[4:0];
        relu_en           = 1'b1;
        act_in            = 16'sd0;
        fifo_rd_en        = 1'b0;
        total_outputs     = 0;
        bit_exact_matches = 0;
        correct_preds     = 0;
        total_inf_cycles  = 0;

        repeat (5) @(negedge clk);
        rst_n = 1'b1;
        repeat (2) @(negedge clk);

        for (img = 0; img < NUM_IMAGES; img = img + 1) begin
            img_start_cycle = cycle_count;

            // Layer 1 (4 passes x 64 inputs)
            for (p = 0; p < 4; p = p + 1) begin
                run_pass(p * 64, 64, cfg_mode_l1[1:0], cfg_shift_l1[4:0], 1'b1, 1'b0, img, fifo_word);
                hidden_act[p*4 + 0] = fifo_word[15:0];
                hidden_act[p*4 + 1] = fifo_word[31:16];
                hidden_act[p*4 + 2] = fifo_word[47:32];
                hidden_act[p*4 + 3] = fifo_word[63:48];
            end

            // Layer 2 (3 passes x 16 inputs) - dynamically switches mode_in & shift_in
            for (p = 0; p < 3; p = p + 1) begin
                run_pass(256 + p * 16, 16, cfg_mode_l2[1:0], cfg_shift_l2[4:0], 1'b0, 1'b1, img, fifo_word);
                rtl_logits[p*4 + 0] = fifo_word[15:0];
                rtl_logits[p*4 + 1] = fifo_word[31:16];
                rtl_logits[p*4 + 2] = fifo_word[47:32];
                rtl_logits[p*4 + 3] = fifo_word[63:48];
            end

            total_inf_cycles = total_inf_cycles + (cycle_count - img_start_cycle);

            max_logit  = rtl_logits[0];
            pred_class = 0;
            for (n = 0; n < 10; n = n + 1) begin
                total_outputs = total_outputs + 1;
                if (rtl_logits[n] === expected_outputs[img*10 + n])
                    bit_exact_matches = bit_exact_matches + 1;

                if (rtl_logits[n] > max_logit) begin
                    max_logit  = rtl_logits[n];
                    pred_class = n;
                end
            end

            if (pred_class == test_labels[img])
                correct_preds = correct_preds + 1;
        end

        $display("========================================================");
        $display("  HIL RESULTS | L1(mode=%0d,sh=%0d) -> L2(mode=%0d,sh=%0d)",
                 cfg_mode_l1, cfg_shift_l1, cfg_mode_l2, cfg_shift_l2);
        $display("========================================================");
        $display("  Bit-Exact Matches    : %0d / %0d", bit_exact_matches, total_outputs);
        $display("  RTL Accuracy         : %0d / %0d (%.2f%%)",
                 correct_preds, NUM_IMAGES, (correct_preds * 100.0) / NUM_IMAGES);
        $display("  Cycles / Inference   : %0d cycles", total_inf_cycles / NUM_IMAGES);
        $display("========================================================");
        $finish;
    end

endmodule