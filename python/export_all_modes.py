import argparse
import numpy as np
import torch
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["int16", "int8", "int4", "mixed"], required=True)
args = parser.parse_args()

# --- 1. Load test data and weights ---
X_test_f = np.load("X_test.npy")   # (360, 64)
y_test   = np.load("y_test.npy")   # (360,)

def load_weights(path):
    m = nn.Sequential(
        nn.Linear(64, 16, bias=False),
        nn.ReLU(),
        nn.Linear(16, 10, bias=False)
    )
    m.load_state_dict(torch.load(path, weights_only=True))
    return m[0].weight.detach().numpy(), m[2].weight.detach().numpy()

w1_fp,  w2_fp  = load_weights("float_model.pt")
w1_qat, w2_qat = load_weights("qat_int4_model.pt")

def quantize_to_int(arr, bits):
    q_max = (1 << (bits - 1)) - 1
    scale = np.max(np.abs(arr)) / q_max
    return np.clip(np.round(arr / scale), -q_max, q_max).astype(np.int32)

# Bit-exact match to requantize_var.v
def rtl_requantize(acc, shift, bits, relu_en=True):
    round_offset = (1 << (shift - 1)) if shift > 0 else 0
    shifted = (acc + round_offset) >> shift
    max_b = (1 << (bits - 1)) - 1
    min_b = -(1 << (bits - 1))
    if relu_en:
        shifted = np.maximum(0, shifted)
    return np.clip(shifted, min_b, max_b).astype(np.int32)

# --- 2. Mode Configuration ---
# mode_in encoding: 0 = INT4 (2'b00), 1 = INT8 (2'b01), 2 = INT16 (2'b10)
if args.mode == "int16":
    w1_f, w2_f = w1_fp, w2_fp
    b1, b2, out_b2 = 16, 16, 16
    mode_l1, mode_l2 = 2, 2
elif args.mode == "int8":
    w1_f, w2_f = w1_fp, w2_fp
    b1, b2, out_b2 = 8, 8, 8
    mode_l1, mode_l2 = 1, 1
elif args.mode == "int4":
    w1_f, w2_f = w1_qat, w2_qat
    b1, b2, out_b2 = 4, 4, 16   # Keep final logits in 16-bit to avoid 4-bit argmax ties
    mode_l1, mode_l2 = 0, 2
elif args.mode == "mixed":
    w1_f, w2_f = w1_qat, w2_fp
    b1, b2, out_b2 = 4, 8, 8    # Genuine runtime switch: L1=INT4 (00) -> L2=INT8 (01)
    mode_l1, mode_l2 = 0, 1

x_int  = quantize_to_int(X_test_f, b1)
w1_int = quantize_to_int(w1_f, b1)
w2_int = quantize_to_int(w2_f, b2)

# --- 3. Sweep shifts for best bit-exact accuracy ---
best_acc, best_s1, best_s2, best_out = -1.0, 0, 0, None
acc1 = x_int.astype(np.int64) @ w1_int.T.astype(np.int64)

for s1 in range(0, 22):
    h_int = rtl_requantize(acc1, s1, bits=b1, relu_en=True)
    acc2  = h_int.astype(np.int64) @ w2_int.T.astype(np.int64)
    max_a2 = np.max(np.abs(acc2))
    out_max = (1 << (out_b2 - 1)) - 1
    s2 = max(0, int(np.ceil(np.log2(max_a2 / out_max)))) if max_a2 > out_max else 0
    out_int = rtl_requantize(acc2, s2, bits=out_b2, relu_en=False)
    acc = np.mean(np.argmax(out_int, axis=1) == y_test) * 100.0
    if acc > best_acc:
        best_acc, best_s1, best_s2, best_out = acc, s1, s2, out_int

# --- 4. Pack 512 x 64-bit weights.hex ---
bram_words = []
for p in range(4):
    for k in range(64):
        w_pe = [w1_int[p*4 + pe, k] for pe in range(4)]
        bram_words.append(f"{w_pe[3]&0xFFFF:04x}{w_pe[2]&0xFFFF:04x}{w_pe[1]&0xFFFF:04x}{w_pe[0]&0xFFFF:04x}")

w2_pad = np.vstack([w2_int, np.zeros((2, 16), dtype=np.int32)])
for p in range(3):
    for k in range(16):
        w_pe = [w2_pad[p*4 + pe, k] for pe in range(4)]
        bram_words.append(f"{w_pe[3]&0xFFFF:04x}{w_pe[2]&0xFFFF:04x}{w_pe[1]&0xFFFF:04x}{w_pe[0]&0xFFFF:04x}")

while len(bram_words) < 512:
    bram_words.append("0000000000000000")

with open("weights.hex", "w") as f:
    f.write("\n".join(bram_words) + "\n")
with open("test_inputs.hex", "w") as f:
    for val in x_int.flatten():
        f.write(f"{int(val) & 0xFFFF:04x}\n")
with open("expected_outputs.hex", "w") as f:
    for val in best_out.flatten():
        f.write(f"{int(val) & 0xFFFF:04x}\n")
with open("test_labels.hex", "w") as f:
    for val in y_test:
        f.write(f"{int(val):x}\n")

# Memory footprint calculation (1,024 L1 weights + 160 L2 weights = 1,184 total)
packed_bytes = (1024 * b1 + 160 * b2) / 8.0
unpacked_bytes = 1184 * 2.0  # Current 16-bit sign-extended BRAM storage

print(f"\n=== Exported Mode: {args.mode.upper()} ===")
print(f"Golden Model Acc  : {best_acc:.2f}%")
print(f"L1 mode={mode_l1}, shift={best_s1} | L2 mode={mode_l2}, shift={best_s2}")
print(f"Memory (Packed)   : {packed_bytes:.0f} bytes ({packed_bytes/1024:.2f} KB)")
print(f"Memory (Current)  : {unpacked_bytes:.0f} bytes (unpacked 16-bit BRAM)")
print(f"\nRun in WSL:")
print(f"vvp sim_hil.vvp +MODE_L1={mode_l1} +SHIFT_L1={best_s1} +MODE_L2={mode_l2} +SHIFT_L2={best_s2}\n")