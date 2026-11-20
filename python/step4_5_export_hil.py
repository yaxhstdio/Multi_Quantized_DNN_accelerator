import numpy as np
import torch
import torch.nn as nn

# --- 1. Load test data and trained INT8 weights ---
X_test_f = np.load("X_test.npy")   # (360, 64)
y_test   = np.load("y_test.npy")   # (360,)

m = nn.Sequential(
    nn.Linear(64, 16, bias=False),
    nn.ReLU(),
    nn.Linear(16, 10, bias=False)
)
m.load_state_dict(torch.load("float_model.pt", weights_only=True))
w1_f = m[0].weight.detach().numpy() # (16, 64)
w2_f = m[2].weight.detach().numpy() # (10, 16)

def quantize_to_int(arr, bits):
    q_max = (1 << (bits - 1)) - 1
    scale = np.max(np.abs(arr)) / q_max
    return np.clip(np.round(arr / scale), -q_max, q_max).astype(np.int32)

# --- 2. Bit-exact match to requantize_var.v ---
def rtl_requantize(acc, shift, bits, relu_en=True):
    # Matches: round_offset = (shift > 0) ? (1 <<< (shift - 1)) : 0
    round_offset = (1 << (shift - 1)) if shift > 0 else 0
    shifted = (acc + round_offset) >> shift

    # Matches: asymmetric min_bound (-8 / -128 / -32768)
    max_b = (1 << (bits - 1)) - 1
    min_b = -(1 << (bits - 1))

    if relu_en:
        shifted = np.maximum(0, shifted)
    return np.clip(shifted, min_b, max_b).astype(np.int32)

# Quantize to INT8 (mode_in = 2'b01)
x_int  = quantize_to_int(X_test_f, 8)
w1_int = quantize_to_int(w1_f, 8)
w2_int = quantize_to_int(w2_f, 8)

SHIFT_L1 = 11
SHIFT_L2 = 0

# Run bit-exact integer inference
acc1 = x_int.astype(np.int64) @ w1_int.T.astype(np.int64)
h_int = rtl_requantize(acc1, SHIFT_L1, bits=8, relu_en=True)

acc2 = h_int.astype(np.int64) @ w2_int.T.astype(np.int64)
out_int = rtl_requantize(acc2, SHIFT_L2, bits=16, relu_en=False)

acc_pct = np.mean(np.argmax(out_int, axis=1) == y_test) * 100.0
print(f"Golden Model (bit-exact to requantize_var.v) Accuracy: {acc_pct:.2f}%")

# --- 3. Pack 512 x 64-bit weights.hex (PE3_PE2_PE1_PE0 per line) ---
# Addr   0..255 : Layer 1 (4 passes x 64 words)
# Addr 256..303 : Layer 2 (3 passes x 16 words, neurons 10-11 zero-padded)
# Addr 304..511 : Zero-padded
bram_words = []

# Layer 1: 4 passes of 4 neurons
for p in range(4):
    for k in range(64):
        w_pe = [w1_int[p*4 + pe, k] for pe in range(4)]
        word = f"{w_pe[3]&0xFFFF:04x}{w_pe[2]&0xFFFF:04x}{w_pe[1]&0xFFFF:04x}{w_pe[0]&0xFFFF:04x}"
        bram_words.append(word)

# Layer 2: 3 passes of 4 neurons (pad neurons 10, 11 with 0)
w2_pad = np.vstack([w2_int, np.zeros((2, 16), dtype=np.int32)])
for p in range(3):
    for k in range(16):
        w_pe = [w2_pad[p*4 + pe, k] for pe in range(4)]
        word = f"{w_pe[3]&0xFFFF:04x}{w_pe[2]&0xFFFF:04x}{w_pe[1]&0xFFFF:04x}{w_pe[0]&0xFFFF:04x}"
        bram_words.append(word)

# Pad remainder to 512 words
while len(bram_words) < 512:
    bram_words.append("0000000000000000")

with open("weights.hex", "w") as f:
    f.write("\n".join(bram_words) + "\n")

# --- 4. Export test inputs, expected outputs, and labels ---
with open("test_inputs.hex", "w") as f:
    for val in x_int.flatten():
        f.write(f"{int(val) & 0xFFFF:04x}\n")

with open("expected_outputs.hex", "w") as f:
    for val in out_int.flatten():
        f.write(f"{int(val) & 0xFFFF:04x}\n")

with open("test_labels.hex", "w") as f:
    for val in y_test:
        f.write(f"{int(val):x}\n")

print("Generated: weights.hex (512x64b), test_inputs.hex, expected_outputs.hex, test_labels.hex")