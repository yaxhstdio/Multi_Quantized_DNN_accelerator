import numpy as np

def quantize_and_clip(acc_val, mode, relu_en):
    """
    Emulates requantize_var.v behavior in Python.
    mode: 'int4', 'int8', 'int16'
    """
    if mode == 'int4':
        shift_bits = 3
        max_bound, min_bound = 7, -8
    elif mode == 'int8':
        shift_bits = 7
        max_bound, min_bound = 127, -128
    elif mode == 'int16':
        shift_bits = 14
        max_bound, min_bound = 32767, -32768
    else:
        raise ValueError("Unknown mode")

    # 1. Rounding offset (half-LSB addition before right shift)
    round_offset = (1 << (shift_bits - 1)) if shift_bits > 0 else 0
    
    # 2. Arithmetic right shift (simulating Verilog '>>>')
    shifted = (acc_val + round_offset) >> shift_bits
    
    # 3. ReLU activation
    if relu_en and shifted < 0:
        return 0
    
    # 4. Saturation clipping
    if shifted > max_bound:
        return max_bound
    elif shifted < min_bound:
        return min_bound
    else:
        return int(shifted)

def run_golden_model():
    print("=== Running Accelerator Golden Model ===")
    
    # Configuration for 4 PEs, INT8 mode
    mode = 'int8'
    relu_en = True
    
    # 1. Generate test activations and weights (4 PEs)
    # 2 cycles of feature inputs
    activations = [32, 64] 
    
    # 4 sets of weights (one for each PE)
    weights = [
        [16, 16],   # PE0: Normal positive dot product
        [32, 32],   # PE1: Larger positive dot product
        [-16, -16], # PE2: Negative dot product (tests ReLU)
        [100, 100]  # PE3: High sum (tests positive saturation)
    ]
    
    biases = [0, 0, 0, 0]
    
    pe_outputs = []
    
    for pe_idx in range(4):
        # Compute dot product: bias + sum(act * weight)
        acc = biases[pe_idx]
        for c in range(len(activations)):
            acc += activations[c] * weights[pe_idx][c]
            
        # Requantize
        quantized = quantize_and_clip(acc, mode, relu_en)
        pe_outputs.append((acc, quantized))
        print(f"PE {pe_idx} -> Accumulator: {acc:6d} | Requantized Output: {quantized:4d}")

    # 2. Generate a sample weights.hex file for Verilog $readmemh
    # Using & 0xFFFF converts negative numbers into valid 16-bit 2's complement hex
    # Each 64-bit line packs 4 16-bit weights: {w3, w2, w1, w0}
    w0_hex = weights[0][0] & 0xFFFF
    w1_hex = weights[1][0] & 0xFFFF
    w2_hex = weights[2][0] & 0xFFFF
    w3_hex = weights[3][0] & 0xFFFF
    hex_line = f"{w3_hex:04x}{w2_hex:04x}{w1_hex:04x}{w0_hex:04x}"
    
    w0_c2 = weights[0][1] & 0xFFFF
    w1_c2 = weights[1][1] & 0xFFFF
    w2_c2 = weights[2][1] & 0xFFFF
    w3_c2 = weights[3][1] & 0xFFFF
    hex_line_c2 = f"{w3_c2:04x}{w2_c2:04x}{w1_c2:04x}{w0_c2:04x}"

    with open("weights.hex", "w") as f:
        f.write(hex_line + "\n")
        f.write(hex_line_c2 + "\n")
        
    print("\nSuccessfully generated 'weights.hex' for Verilog BRAM simulation!")

if __name__ == "__main__":
    run_golden_model()