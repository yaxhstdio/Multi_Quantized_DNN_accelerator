`timescale 1ns / 1ps

module mac_control_unit #(
    parameter ADDR_WIDTH = 6
)(
    input  wire                  clk,
    input  wire                  rst_n,
    input  wire                  start,         // Trigger to start execution
    input  wire [ADDR_WIDTH-1:0] layer_length,  // Number of input features per neuron (e.g., 64)
    input  wire                  valid_out,     // From MAC array (signals computation complete)
    
    // Outputs to BRAM & Memory
    output reg                   bram_rd_en,    // Read enable for Weight BRAM
    output reg  [ADDR_WIDTH-1:0] bram_rd_addr,  // Address pointer for BRAM
    output reg                   fifo_wr_en,    // Write enable for Activation FIFO
    
    // Outputs to MAC Array
    output reg                   mac_en,        // Enables PE multiplier/adder
    output reg                   accum_clr,     // Clears/initializes accumulator on cycle 0
    output reg                   last_cycle     // Signals final cycle of dot-product
);

    // State Encoding
    localparam S_IDLE      = 2'b00;
    localparam S_FETCH     = 2'b01;
    localparam S_COMPUTE   = 2'b10;
    localparam S_WRITEBACK = 2'b11;

    reg [1:0] state, next_state;
    reg [ADDR_WIDTH-1:0] feature_counter;

    // State Register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= S_IDLE;
            feature_counter <= 0;
        end else begin
            state <= next_state;
            
            // Feature counter logic during compute state
            if (state == S_IDLE) begin
                feature_counter <= 0;
            end else if (state == S_COMPUTE) begin
                if (feature_counter < layer_length - 1)
                    feature_counter <= feature_counter + 1'b1;
                else
                    feature_counter <= 0;
            end
        end
    end

    // Next State & Control Logic FSM
    always @(*) begin
        // Default control signal values
        next_state   = state;
        bram_rd_en   = 1'b0;
        bram_rd_addr = feature_counter;
        fifo_wr_en   = 1'b0;
        mac_en       = 1'b0;
        accum_clr    = 1'b0;
        last_cycle   = 1'b0;

        case (state)
            S_IDLE: begin
                if (start)
                    next_state = S_FETCH;
            end

            S_FETCH: begin
                bram_rd_en  = 1'b1;
                next_state  = S_COMPUTE;
            end

            S_COMPUTE: begin
                mac_en      = 1'b1;
                bram_rd_en  = 1'b1; // Fetch next weight simultaneously
                
                // Assert accum_clr only on the very first feature cycle
                if (feature_counter == 0) begin
                    accum_clr = 1'b1;
                end

                // Assert last_cycle on the final feature cycle
                if (feature_counter == layer_length - 1) begin
                    last_cycle = 1'b1;
                    next_state = S_WRITEBACK;
                end
            end

            S_WRITEBACK: begin
                if (valid_out) begin
                    fifo_wr_en = 1'b1;
                    next_state = S_IDLE; // Task complete or loop to next layer
                end else begin
                    next_state = S_WRITEBACK;
                end
            end

            default: next_state = S_IDLE;
        endcase
    end

endmodule