`timescale 1ns / 1ps

module accelerator_top #(
    parameter NUM_PES        = 4,
    parameter MAX_DATA_WIDTH = 16,
    parameter ACC_WIDTH      = 32,
    parameter ADDR_WIDTH     = 6
)(
    // --------------------------------------------------------
    // Physical I/O Pins of the Chip
    // --------------------------------------------------------
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              start,
    input  wire [ADDR_WIDTH-1:0]             layer_length,
    input  wire [1:0]                        mode_in,
    input  wire                              relu_en,
    input  wire signed [MAX_DATA_WIDTH-1:0]  act_in,      // Feature streamed into the chip
    input  wire                              fifo_rd_en,  // External read signal

    output wire [MAX_DATA_WIDTH-1:0]         fifo_rd_data, // Final output leaving the chip
    output wire                              fifo_full,
    output wire                              fifo_empty
);

    // --------------------------------------------------------
    // Internal Copper Traces (Wires)
    // --------------------------------------------------------
    wire                               bram_rd_en;
    wire [ADDR_WIDTH-1:0]              bram_rd_addr;
    wire [63:0]                        bram_rd_data;
    
    wire                               fifo_wr_en;
    wire                               mac_en;
    wire                               accum_clr;
    wire                               last_cycle;
    wire                               valid_out;
    
    wire [(NUM_PES*ACC_WIDTH)-1:0]      acc_out_flat;
    wire [(NUM_PES*MAX_DATA_WIDTH)-1:0] quant_out_flat;
    
    // Tying biases to zero internally to save I/O pin count during synthesis
    wire [(NUM_PES*ACC_WIDTH)-1:0]      biases_flat = 0;

    // --------------------------------------------------------
    // 1. Parameter BRAM
    // --------------------------------------------------------
    parameter_bram #(
        .DATA_WIDTH (64),
        .ADDR_WIDTH (ADDR_WIDTH)
    ) u_bram (
        .clk        (clk),
        .rd_en      (bram_rd_en),
        .rd_addr    (bram_rd_addr),
        .rd_data    (bram_rd_data)
    );

    // --------------------------------------------------------
    // 2. Master Control Unit (FSM)
    // --------------------------------------------------------
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

    // --------------------------------------------------------
    // 3. Configurable MAC Array
    // --------------------------------------------------------
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
        .weights_flat   (bram_rd_data),
        .biases_flat    (biases_flat),
        .acc_out_flat   (acc_out_flat),
        .quant_out_flat (quant_out_flat),
        .valid_out      (valid_out)
    );

    // --------------------------------------------------------
    // 4. Activation FIFO
    // --------------------------------------------------------
    sync_fifo #(
        .DATA_WIDTH (MAX_DATA_WIDTH),
        .ADDR_WIDTH (4)
    ) u_fifo (
        .clk        (clk),
        .rst_n      (rst_n),
        .wr_en      (fifo_wr_en),
        .wr_data    (quant_out_flat[15:0]), // Routing PE0's requantized output to FIFO
        .rd_en      (fifo_rd_en),
        .rd_data    (fifo_rd_data),
        .full       (fifo_full),
        .empty      (fifo_empty)
    );

endmodule
