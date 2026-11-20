`timescale 1ns / 1ps

module parameter_bram #(
    parameter DATA_WIDTH = 64, // Width of packed weights for our 4 PEs (4 x 16 bits)
    parameter ADDR_WIDTH = 6   // Depth = 2^6 = 64 address lines
)(
    input  wire                    clk,
    input  wire                    rd_en,     // Read enable from control unit
    input  wire [ADDR_WIDTH-1:0]   rd_addr,   // Memory address pointer
    output reg  [DATA_WIDTH-1:0]   rd_data    // 64-bit packed weights output
);

    // BRAM array declaration
    reg [DATA_WIDTH-1:0] bram [0:(1<<ADDR_WIDTH)-1];

    // Initialize memory from a text file during simulation startup
    initial begin
        $readmemh("weights.hex", bram);
    end

    // Synchronous Read
    always @(posedge clk) begin
        if (rd_en) begin
            rd_data <= bram[rd_addr];
        end
    end

endmodule