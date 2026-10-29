`timescale 1ns / 1ps

module tb_accelerator_top;
    reg clk;
    reg rst_n;
    reg start;
    reg [5:0] layer_length;
    reg [1:0] mode_in;
    reg relu_en;
    reg signed [15:0] act_in;
    reg fifo_rd_en;

    wire [15:0] fifo_rd_data;
    wire fifo_full;
    wire fifo_empty;

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

    initial clk = 0;
    always #5 clk = ~clk;

    initial begin
        $dumpfile("tb_accelerator_top.vcd");
        $dumpvars(0, tb_accelerator_top);

        rst_n = 0; start = 0; layer_length = 2; 
        mode_in = 2'b01; relu_en = 1; act_in = 0; fifo_rd_en = 0;

        #20 rst_n = 1;
        #10;

        // Feed test features
        @(negedge clk);
        start = 1; act_in = 16'sd32; 

        @(negedge clk);
        start = 0; act_in = 16'sd64; 

        // Wait for pipeline to finish
        #80;

        // Pop data from FIFO
        @(negedge clk); fifo_rd_en = 1; 
        @(negedge clk); fifo_rd_en = 0;

        #20;
        $display("========================================");
        $display("   TOP WRAPPER SIMULATION RESULTS       ");
        $display("========================================");
        $display("FIFO Empty : %0b", fifo_empty);
        $display("FIFO Data  : %0d", $signed(fifo_rd_data));
        $display("========================================");
        $finish;
    end
endmodule
