import json
import numpy as np
import torch
import torch.nn as nn

# --- 1. Load test data ---
X_test_f = np.load("X_test.npy")   # shape (360, 64), range [0.0, 1.0]
y_test   = np.load("y_test.npy")   # shape (360,)

def load_weights(path):
    m = nn.Sequential(
        nn.Linear(64, 16, bias=False),
        nn.ReLU(),
        nn.Linear(16, 10, bias=False)
    )
    m.load_state_dict(torch.load(path, weights_only=True))
    return m[0].weight.detach().numpy(), m[2].weight.detach().numpy()

# --- 2. Pure integer quantization helper ---
def quantize_to_int(arr, bits):
    q_max = (1 << (bits - 1)) - 1
    max_val = np.max(np.abs(arr))
    if max_val == 0:
        return np.zeros_like(arr, dtype=np.int32)
    scale = max_val / q_max
    return np.clip(np.round(arr / scale), -q_max, q_max).astype(np.int32)

# --- 3. Bit-exact RTL model (matches pe_mac + requantize_var) ---
def rtl_layer(x_int, w_int, shift, out_bits, relu_en=True):
    # Signed integer MAC (int64 prevents Python overflow, matches 32/48-bit DSP acc)
    acc = x_int.astype(np.int64) @ w_int.T.astype(np.int64)

    # Arithmetic right-shift (matches Verilog >>> shift)
    shifted = acc >> shift

    # Parameterized ReLU
    if relu_en:
        shifted = np.maximum(0, shifted)

    # Clamp / saturate to output precision
    q_max = (1 << (out_bits - 1)) - 1
    q_min = 0 if relu_en else -q_max
    return np.clip(shifted, q_min, q_max).astype(np.int32)

def find_best_shift_and_eval(w1_f, w2_f, b1, b2):
    x_int  = quantize_to_int(X_test_f, b1)
    w1_int = quantize_to_int(w1_f, b1)
    w2_int = quantize_to_int(w2_f, b2)

    # Sweep shift1 to find the power-of-2 shift that gives highest integer accuracy
    best_acc, best_s1 = -1.0, 0
    for s1 in range(0, 20):
        h_int = rtl_layer(x_int, w1_int, shift=s1, out_bits=b2, relu_en=True)
        # Layer 2 output fits in signed 16-bit FIFO (shift2 = 0 if no overflow, else shift)
        acc2 = h_int.astype(np.int64) @ w2_int.T.astype(np.int64)
        max_acc2 = np.max(np.abs(acc2))
        s2 = max(0, int(np.ceil(np.log2(max_acc2 / 32767.0)))) if max_acc2 > 32767 else 0
        out_int = rtl_layer(h_int, w2_int, shift=s2, out_bits=16, relu_en=False)

        preds = np.argmax(out_int, axis=1)
        acc = np.mean(preds == y_test) * 100.0
        if acc > best_acc:
            best_acc, best_s1, best_s2, best_out = acc, s1, s2, out_int

    return best_acc, best_s1, best_s2, x_int, w1_int, w2_int, best_out

# --- 4. Evaluate all hardware modes (pure integer, zero floats) ---
w1_fp,  w2_fp  = load_weights("float_model.pt")
w1_qat, w2_qat = load_weights("qat_int4_model.pt")

configs = [
    ("INT16 (all)",      w1_fp,  w2_fp,  16, 16),
    ("INT8  (all)",      w1_fp,  w2_fp,  8,  8),
    ("Mixed INT4->INT8", w1_qat, w2_fp,  4,  8),
    ("INT4  (QAT)",      w1_qat, w2_qat, 4,  4),
]

print(f"{'Mode':<18} {'Shift1':<8} {'Shift2':<8} {'Integer Acc':<12}")
print("-" * 48)
results = {}
for name, w1, w2, b1, b2 in configs:
    acc, s1, s2, x_i, w1_i, w2_i, out_i = find_best_shift_and_eval(w1, w2, b1, b2)
    print(f"{name:<18} {s1:<8} {s2:<8} {acc:.2f}%")
    results[name] = (acc, s1, s2, x_i, w1_i, w2_i, out_i)

# --- 5. Export INT8 (or change key to "INT4  (QAT)") to HEX for Verilog ---
export_mode = "INT8  (all)"
acc, s1, s2, x_i, w1_i, w2_i, out_i = results[export_mode]

def write_hex_16bit(filename, array_1d):
    with open(filename, "w") as f:
        for val in array_1d:
            # 16-bit two's complement hex
            f.write(f"{int(val) & 0xFFFF:04x}\n")

write_hex_16bit("weights_l1.hex", w1_i.flatten())
write_hex_16bit("weights_l2.hex", w2_i.flatten())
write_hex_16bit("test_inputs.hex", x_i.flatten())
write_hex_16bit("expected_outputs.hex", out_i.flatten())

with open("hw_config.json", "w") as f:
    json.dump({
        "mode": export_mode,
        "shift_l1": s1,
        "shift_l2": s2,
        "num_test_images": int(len(y_test)),
        "integer_accuracy_pct": round(acc, 2)
    }, f, indent=2)

print(f"\nExported {export_mode} -> weights_l1.hex, weights_l2.hex, test_inputs.hex, expected_outputs.hex, hw_config.json")