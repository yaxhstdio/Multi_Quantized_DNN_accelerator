module mac_configurable (
    input clk,
    input reset,
    input [1:0] mode,        // 00=INT4, 01=INT8, 10=INT16
    input signed [15:0] a,
    input signed [15:0] b,
    output reg signed [33:0] acc
);

endmodule