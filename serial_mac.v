module serial_mac #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 24
)(
    input  wire                         clk,
    input  wire                         rst_n,      // Active-low synchronous/async reset
    input  wire                         mac_en,     // Enable accumulation
    input  wire                         accum_clr,  // 1: Start new dot product (load bias + prod)
    input  wire signed [DATA_WIDTH-1:0] act_in,     // Signed INT8 Activation
    input  wire signed [DATA_WIDTH-1:0] weight_in,  // Signed INT8 Weight
    input  wire signed [ACC_WIDTH-1:0]  bias_in,    // Signed INT24 Bias (or 0 if unused)
    output reg  signed [ACC_WIDTH-1:0]  accum_out   // Signed INT24 Accumulated Output
);

    // Full-precision signed multiplication (2 * DATA_WIDTH = 16 bits)
    wire signed [(2*DATA_WIDTH)-1:0] product;
    assign product = act_in * weight_in;

    // Sign-extend product to ACC_WIDTH (24 bits)
    wire signed [ACC_WIDTH-1:0] product_ext;
    assign product_ext = {{(ACC_WIDTH - 2*DATA_WIDTH){product[(2*DATA_WIDTH)-1]}}, product};

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum_out <= {ACC_WIDTH{1'b0}};
        end else if (mac_en) begin
            if (accum_clr) begin
                // First cycle of dot product: initialize with bias + first product
                accum_out <= bias_in + product_ext;
            end else begin
                // Subsequent cycles: accumulate
                accum_out <= accum_out + product_ext;
            end
        end
    end

endmodule