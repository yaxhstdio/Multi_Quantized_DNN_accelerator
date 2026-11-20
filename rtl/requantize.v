module requantize #(
    parameter ACC_WIDTH  = 24,
    parameter OUT_WIDTH  = 8,
    parameter SHIFT_BITS = 7    // Default fixed-point scaling factor (divide by 128)
)(
    input  wire                         relu_en,    // 1: Apply ReLU, 0: Linear pass-through
    input  wire signed [ACC_WIDTH-1:0]  accum_in,   // 24-bit raw accumulator output
    output reg  signed [OUT_WIDTH-1:0]  quant_out   // 8-bit requantized & clipped output
);

    // Maximum and minimum representable values in signed OUT_WIDTH (INT8: +127 to -128)
    localparam signed [ACC_WIDTH-1:0] MAX_VAL =  (1 << (OUT_WIDTH - 1)) - 1; // +127
    localparam signed [ACC_WIDTH-1:0] MIN_VAL = -(1 << (OUT_WIDTH - 1));     // -128

    // Step 1: Optional Half-LSB rounding before arithmetic right-shift
    wire signed [ACC_WIDTH-1:0] rounded_val;
    wire signed [ACC_WIDTH-1:0] shifted_val;
    
    assign rounded_val = accum_in + (1 << (SHIFT_BITS - 1));
    assign shifted_val = rounded_val >>> SHIFT_BITS;

    // Step 2: ReLU & Saturation Clipping
    always @(*) begin
        if (relu_en && (shifted_val < 0)) begin
            // ReLU clamps negative numbers to 0
            quant_out = {OUT_WIDTH{1'b0}};
        end else if (shifted_val > MAX_VAL) begin
            // Positive saturation (+127)
            quant_out = MAX_VAL[OUT_WIDTH-1:0];
        end else if (shifted_val < MIN_VAL) begin
            // Negative saturation (-128, only reachable if relu_en == 0)
            quant_out = MIN_VAL[OUT_WIDTH-1:0];
        end else begin
            // Within valid INT8 range
            quant_out = shifted_val[OUT_WIDTH-1:0];
        end
    end

endmodule