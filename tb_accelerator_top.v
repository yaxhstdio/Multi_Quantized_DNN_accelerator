`timescale 1ns / 1ps

module tb_accelerator_top;

    // Simulated Inputs to the Chip
    reg clk;
    reg rst_n;
    reg start;
    reg [5:0] layer_length;
    reg [1:0] mode_in;
    reg relu_en;
    reg signed [15:0] act_in;
    reg fifo_rd_en;

    // Simulated Outputs from the Chip
    wire [15:0] fifo_rd_data;
    wire fifo_full;
    wire fifo_empty;

    // Instantiate the Top-Level Wrapper
    accelerator_top uut (
        .clk(clk),
        .rst_n(rst_n),
        .start(start),
        .layer_length(layer_length),
        .mode_in(mode_in),
        .relu_en(relu_en),
        .act_in(act_in),
        .fifo_rd_en(fifo_rd_en),
        .fifo_rd_data(fifo_rd_data),
        .fifo_full(fifo_full),
        .fifo_empty(fifo_empty)
    );

    // Generate 100MHz Clock
    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        $dumpfile("tb_accelerator_top.vcd");
        $dumpvars(0, tb_accelerator_top);

        // Initialize pins
        rst_n = 0;
        start = 0;
        layer_length = 2; // Matches the python golden model test
        mode_in = 2'b01;  // INT8 mode
        relu_en = 1;
        act_in = 0;
        fifo_rd_en = 0;

        #20 rst_n = 1;
        #10;

        // -----------------------------------------------------
        // Feed Data into the Chip
        // -----------------------------------------------------
        @(negedge clk);
        start = 1;
        act_in = 16'sd32; // Feature 0

        @(negedge clk);
        start = 0;
        act_in = 16'sd64; // Feature 1

        // Wait for FSM to complete computation and write to FIFO
        #80;

        // -----------------------------------------------------
        // Read Result from the Chip's Output Pin
        // -----------------------------------------------------
        @(negedge clk);
        fifo_rd_en = 1; // Assert Read Enable
        
        @(negedge clk);
        fifo_rd_en = 0; // De-assert Read Enable

        #20;
        $display("========================================");
        $display("   ACCELERATOR TOP WRAPPER SIMULATION   ");
        $display("========================================");
        $display("FIFO Empty Flag : %0b", fifo_empty);
        $display("FIFO Read Data  : %0d", $signed(fifo_rd_data));
        $display("========================================");

        $finish;
    end
endmodule
