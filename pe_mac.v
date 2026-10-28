`timescale 1ns / 1ps

module pe_mac #(
    parameter MAX_DATA_WIDTH = 16,
    parameter ACC_WIDTH      = 32
)(
    input  wire                              clk,
    input  wire                              rst_n,
    input  wire                              mac_en,
    input  wire                              accum_clr,
    input  wire                              is_int4,
    input  wire                              is_int8,
    input  wire signed [MAX_DATA_WIDTH-1:0]  act_in,
    input  wire signed [MAX_DATA_WIDTH-1:0]  weight_in,
    input  wire signed [ACC_WIDTH-1:0]       bias_in,
    output reg  signed [ACC_WIDTH-1:0]       accum_out
);

    // Sign-extend inputs dynamically based on active precision mode
    reg signed [MAX_DATA_WIDTH-1:0] act_masked;
    reg signed [MAX_DATA_WIDTH-1:0] weight_masked;

    always @(*) begin
        if (is_int4) begin
            act_masked    = {{12{act_in[3]}}, act_in[3:0]};
            weight_masked = {{12{weight_in[3]}}, weight_in[3:0]};
        end else if (is_int8) begin
            act_masked    = {{8{act_in[7]}}, act_in[7:0]};
            weight_masked = {{8{weight_in[7]}}, weight_in[7:0]};
        end else begin // INT16
            act_masked    = act_in;
            weight_masked = weight_in;
        end
    end

    // Multiply and accumulate into 32-bit register
    wire signed [(2*MAX_DATA_WIDTH)-1:0] product;
    assign product = act_masked * weight_masked;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            accum_out <= {ACC_WIDTH{1'b0}};
        end else if (mac_en) begin
            if (accum_clr)
                accum_out <= bias_in + product;
            else
                accum_out <= accum_out + product;
        end
    end

endmodule