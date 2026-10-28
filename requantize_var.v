`timescale 1ns / 1ps

module requantize_var #(
    parameter ACC_WIDTH      = 32,
    parameter MAX_DATA_WIDTH = 16
)(
    input  wire                              relu_en,
    input  wire                              is_int4,
    input  wire                              is_int8,
    input  wire [4:0]                        shift_bits,
    input  wire signed [ACC_WIDTH-1:0]       accum_in,
    output reg  signed [MAX_DATA_WIDTH-1:0]  quant_out
);

    // Dynamic rounding and arithmetic right shift
    wire signed [ACC_WIDTH-1:0] round_offset;
    wire signed [ACC_WIDTH-1:0] shifted_val;

    assign round_offset = (shift_bits > 0) ? (32'sd1 <<< (shift_bits - 1'b1)) : 32'sd0;
    assign shifted_val  = (accum_in + round_offset) >>> shift_bits;

    // Dynamic Saturation Bounds based on mode
    reg signed [ACC_WIDTH-1:0] max_bound;
    reg signed [ACC_WIDTH-1:0] min_bound;

    always @(*) begin
        if (is_int4) begin
            max_bound = 32'sd7;
            min_bound = -32'sd8;
        end else if (is_int8) begin
            max_bound = 32'sd127;
            min_bound = -32'sd128;
        end else begin
            max_bound = 32'sd32767;
            min_bound = -32'sd32768;
        end
    end

    // ReLU + Saturation Clipping
    always @(*) begin
        if (relu_en && (shifted_val < 0)) begin
            quant_out = {MAX_DATA_WIDTH{1'b0}};
        end else if (shifted_val > max_bound) begin
            quant_out = max_bound[MAX_DATA_WIDTH-1:0];
        end else if (shifted_val < min_bound) begin
            quant_out = min_bound[MAX_DATA_WIDTH-1:0];
        end else begin
            quant_out = shifted_val[MAX_DATA_WIDTH-1:0];
        end
    end

endmodule