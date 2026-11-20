`timescale 1ns / 1ps

module tb_single_infer;

    localparam ADDR_WIDTH = 9;

    reg                  clk, rst_n, start, relu_en, fifo_rd_en;
    reg [ADDR_WIDTH-1:0] base_addr, layer_length;
    reg [1:0]            mode_in;
    reg [4:0]            shift_in;
    reg signed [15:0]    act_in;

    wire [63:0]          fifo_rd_data;
    wire                 fifo_full, fifo_empty;

    reg signed [15:0]    img_in     [0:63];
    reg signed [15:0]    hidden_act [0:15];
    reg signed [15:0]    rtl_logits [0:11];

    integer cfg_mode_l1, cfg_shift_l1, cfg_mode_l2, cfg_shift_l2;
    integer cycle_count, start_cycle;

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
        input [ADDR_WIDTH-1:0] t_base, t_len;
        input [1:0]            t_mode;
        input [4:0]            t_shift;
        input                  t_relu, is_layer2;
        output [63:0]          pass_out;
        integer k;
        begin
            @(negedge clk);
            base_addr = t_base; layer_length = t_len;
            mode_in = t_mode; shift_in = t_shift; relu_en = t_relu;
            start = 1'b1;
            @(negedge clk);
            start = 1'b0;

            k = 0;
            while (k < t_len) begin
                if (u_dut.mac_en) begin
                    act_in = is_layer2 ? hidden_act[k] : img_in[k];
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

    integer p, n, pred_class;
    reg signed [15:0] max_logit;
    reg [63:0] fifo_word;

    initial begin
        cfg_mode_l1 = 1; cfg_shift_l1 = 11;
        cfg_mode_l2 = 1; cfg_shift_l2 = 7;
        if (!$value$plusargs("MODE_L1=%d",  cfg_mode_l1))  ;
        if (!$value$plusargs("SHIFT_L1=%d", cfg_shift_l1)) ;
        if (!$value$plusargs("MODE_L2=%d",  cfg_mode_l2))  ;
        if (!$value$plusargs("SHIFT_L2=%d", cfg_shift_l2)) ;

        $readmemh("single_input.hex", img_in);

        rst_n = 0; start = 0; base_addr = 0; layer_length = 0;
        mode_in = 0; shift_in = 0; relu_en = 0; act_in = 0; fifo_rd_en = 0;

        repeat (5) @(negedge clk);
        rst_n = 1;
        repeat (2) @(negedge clk);

        start_cycle = cycle_count;

        for (p = 0; p < 4; p = p + 1) begin
            run_pass(p * 64, 64, cfg_mode_l1[1:0], cfg_shift_l1[4:0], 1'b1, 1'b0, fifo_word);
            hidden_act[p*4 + 0] = fifo_word[15:0];
            hidden_act[p*4 + 1] = fifo_word[31:16];
            hidden_act[p*4 + 2] = fifo_word[47:32];
            hidden_act[p*4 + 3] = fifo_word[63:48];
        end

        for (p = 0; p < 3; p = p + 1) begin
            run_pass(256 + p * 16, 16, cfg_mode_l2[1:0], cfg_shift_l2[4:0], 1'b0, 1'b1, fifo_word);
            rtl_logits[p*4 + 0] = fifo_word[15:0];
            rtl_logits[p*4 + 1] = fifo_word[31:16];
            rtl_logits[p*4 + 2] = fifo_word[47:32];
            rtl_logits[p*4 + 3] = fifo_word[63:48];
        end

        max_logit = rtl_logits[0]; pred_class = 0;
        for (n = 0; n < 10; n = n + 1) begin
            if (rtl_logits[n] > max_logit) begin
                max_logit = rtl_logits[n];
                pred_class = n;
            end
        end

        $display("LOGITS:%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d,%0d",
                 rtl_logits[0], rtl_logits[1], rtl_logits[2], rtl_logits[3], rtl_logits[4],
                 rtl_logits[5], rtl_logits[6], rtl_logits[7], rtl_logits[8], rtl_logits[9]);
        $display("PRED:%0d", pred_class);
        $display("CYCLES:%0d", cycle_count - start_cycle);
        $finish;
    end
endmodule