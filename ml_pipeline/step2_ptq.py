import numpy as np
import torch
import torch.nn as nn

# --- 1. Load saved float model and test data ---
model = nn.Sequential(
    nn.Linear(64, 16, bias=False),
    nn.ReLU(),
    nn.Linear(16, 10, bias=False)
)
model.load_state_dict(torch.load("float_model.pt", weights_only=True))
model.eval()

X_test = torch.tensor(np.load("X_test.npy"), dtype=torch.float32)
y_test = torch.tensor(np.load("y_test.npy"), dtype=torch.long)

# --- 2. Symmetric quantize + dequantize (simulates integer precision) ---
def fake_quantize(tensor, bits):
    q_max = (1 << (bits - 1)) - 1
    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor
    scale = max_val / q_max
    q = torch.clamp(torch.round(tensor / scale), -q_max, q_max)
    return q * scale

# --- 3. Evaluate at a given bit-width (weights + activations) ---
def eval_ptq(bits=None):
    with torch.no_grad():
        w1 = model[0].weight.data
        w2 = model[2].weight.data
        x  = X_test

        if bits is not None:
            x  = fake_quantize(x, bits)
            w1 = fake_quantize(w1, bits)
            w2 = fake_quantize(w2, bits)

        # Layer 1 + ReLU
        h = torch.relu(x @ w1.T)
        if bits is not None:
            h = fake_quantize(h, bits)

        # Layer 2
        logits = h @ w2.T
        preds = logits.argmax(dim=1)
        acc = (preds == y_test).float().mean().item()
        return acc * 100.0

# --- 4. Run sweep ---
print(f"FP32  baseline : {eval_ptq(None):.2f}%")
print(f"INT16 PTQ      : {eval_ptq(16):.2f}%")
print(f"INT8  PTQ      : {eval_ptq(8):.2f}%")
print(f"INT4  PTQ      : {eval_ptq(4):.2f}%")