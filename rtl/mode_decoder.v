`timescale 1ns / 1ps

module mode_decoder (
    input  wire [1:0] mode_in,      // 00: INT4, 01: INT8, 10: INT16
    output reg        is_int4,
    output reg        is_int8,
    output reg        is_int16,
    output reg  [4:0] shift_bits    // Dynamic scaling factor for Requantizer
);

    always @(*) begin
        // Default values to prevent latches
        is_int4    = 1'b0;
        is_int8    = 1'b0;
        is_int16   = 1'b0;
        shift_bits = 5'd7;

        case (mode_in)
            2'b00: begin // INT4 Mode ( inputs in [-8, +7] )
                is_int4    = 1'b1;
                shift_bits = 5'd3;  // Divide by 8
            end
            2'b01: begin // INT8 Mode ( inputs in [-128, +127] )
                is_int8    = 1'b1;
                shift_bits = 5'd7;  // Divide by 128
            end
            2'b10: begin // INT16 Mode ( inputs in [-32768, +32767] )
                is_int16   = 1'b1;
                shift_bits = 5'd14; // Divide by 16384
            end
            default: begin // Safe fallback to INT8
                is_int8    = 1'b1;
                shift_bits = 5'd7;
            end
        endcase
    end

endmodule