`timescale 1ns / 1ps

module mac_array_requantized #(
    parameter NUM_PES        = 4,
    parameter MAX_DATA_WIDTH = 16,
    parameter ACC_WIDTH      = 32
)(
    input  wire                                      clk,
    input  wire                                      rst_n,
    input  wire [1:0]                                mode_in,       // 00:INT4, 01:INT8, 10:INT16
    input  wire                                      mac_en,
    input  wire                                      accum_clr,
    input  wire                                      relu_en,
    input  wire                                      last_cycle,    // Asserted on final cycle of dot-product
    input  wire signed [MAX_DATA_WIDTH-1:0]          act_in,        // Broadcasted activation to all 4 PEs
    input  wire [(NUM_PES*MAX_DATA_WIDTH)-1:0]       weights_flat,  // 4 packed weights
    input  wire [(NUM_PES*ACC_WIDTH)-1:0]            biases_flat,   // 4 packed biases
    output wire [(NUM_PES*ACC_WIDTH)-1:0]            acc_out_flat,  // Raw 32-bit accumulators
    output wire [(NUM_PES*MAX_DATA_WIDTH)-1:0]       quant_out_flat,// Requantized outputs
    output reg                                       valid_out      // 1 when dot-product is finished
);

    // 1. Instantiate Mode Decoder
    wire       is_int4, is_int8, is_int16;
    wire [4:0] shift_bits;

    mode_decoder u_decoder (
        .mode_in    (mode_in),
        .is_int4    (is_int4),
        .is_int8    (is_int8),
        .is_int16   (is_int16),
        .shift_bits (shift_bits)
    );

    // 2. Generate 4 Parallel PEs + Requantizers
    genvar i;
    generate
        for (i = 0; i < NUM_PES; i = i + 1) begin : gen_pe_cluster
            
            wire signed [ACC_WIDTH-1:0]      pe_acc_wire;
            wire signed [MAX_DATA_WIDTH-1:0] pe_quant_wire;

            pe_mac #(
                .MAX_DATA_WIDTH(MAX_DATA_WIDTH),
                .ACC_WIDTH     (ACC_WIDTH)
            ) u_pe (
                .clk       (clk),
                .rst_n     (rst_n),
                .mac_en    (mac_en),
                .accum_clr (accum_clr),
                .is_int4   (is_int4),
                .is_int8   (is_int8),
                .act_in    (act_in),
                .weight_in (weights_flat[(i+1)*MAX_DATA_WIDTH - 1 : i*MAX_DATA_WIDTH]),
                .bias_in   (biases_flat[(i+1)*ACC_WIDTH - 1 : i*ACC_WIDTH]),
                .accum_out (pe_acc_wire)
            );

            requantize_var #(
                .ACC_WIDTH     (ACC_WIDTH),
                .MAX_DATA_WIDTH(MAX_DATA_WIDTH)
            ) u_req (
                .relu_en    (relu_en),
                .is_int4    (is_int4),
                .is_int8    (is_int8),
                .shift_bits (shift_bits),
                .accum_in   (pe_acc_wire),
                .quant_out  (pe_quant_wire)
            );

            assign acc_out_flat[(i+1)*ACC_WIDTH - 1 : i*ACC_WIDTH]          = pe_acc_wire;
            assign quant_out_flat[(i+1)*MAX_DATA_WIDTH - 1 : i*MAX_DATA_WIDTH] = pe_quant_wire;

        end
    endgenerate

    // 3. Valid Output Flag Register (High one cycle after the final MAC operation)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            valid_out <= 1'b0;
        else
            valid_out <= (mac_en && last_cycle);
    end

endmodule