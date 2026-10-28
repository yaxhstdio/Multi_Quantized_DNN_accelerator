`timescale 1ns / 1ps

module sync_fifo #(
    parameter DATA_WIDTH = 16, // Width of one data element
    parameter ADDR_WIDTH = 4   // Depth = 2^4 = 16 entries
)(
    input  wire                    clk,
    input  wire                    rst_n,
    input  wire                    wr_en,     // Write enable (from requantizer)
    input  wire [DATA_WIDTH-1:0]   wr_data,   // Data entering the FIFO
    input  wire                    rd_en,     // Read enable (from next layer)
    output reg  [DATA_WIDTH-1:0]   rd_data,   // Data exiting the FIFO
    output wire                    full,      // FIFO is full flag
    output wire                    empty      // FIFO is empty flag
);

    // Memory storage array
    reg [DATA_WIDTH-1:0] mem [0:(1<<ADDR_WIDTH)-1];
    
    // Write and read pointers (extra bit used to distinguish full vs empty)
    reg [ADDR_WIDTH:0]   wr_ptr, rd_ptr;

    assign full  = (wr_ptr == {~rd_ptr[ADDR_WIDTH], rd_ptr[ADDR_WIDTH-1:0]});
    assign empty = (wr_ptr == rd_ptr);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wr_ptr <= 0;
            rd_ptr <= 0;
        end else begin
            // Write operation
            if (wr_en && !full) begin
                mem[wr_ptr[ADDR_WIDTH-1:0]] <= wr_data;
                wr_ptr <= wr_ptr + 1'b1;
            end
            // Read operation
            if (rd_en && !empty) begin
                rd_data <= mem[rd_ptr[ADDR_WIDTH-1:0]];
                rd_ptr <= rd_ptr + 1'b1;
            end
        end
    end

endmodule