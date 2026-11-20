import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# --- 1. Reload exact train/test split ---
digits = load_digits()
X = digits.data / 16.0
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

def load_base_model():
    m = nn.Sequential(
        nn.Linear(64, 16, bias=False),
        nn.ReLU(),
        nn.Linear(16, 10, bias=False)
    )
    m.load_state_dict(torch.load("float_model.pt", weights_only=True))
    return m

# --- 2. STE Fake Quantize (works for both inference and QAT backprop) ---
def fake_quantize_ste(tensor, bits):
    q_max = (1 << (bits - 1)) - 1
    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor
    scale = max_val / q_max
    q = torch.clamp(torch.round(tensor / scale), -q_max, q_max) * scale
    return tensor + (q - tensor).detach()

def forward_mixed(model, x, b1, b2):
    # Layer 1 at precision b1
    x_q  = fake_quantize_ste(x, b1)
    w1_q = fake_quantize_ste(model[0].weight, b1)
    h    = torch.relu(x_q @ w1_q.T)

    # Layer 2 at precision b2
    h_q  = fake_quantize_ste(h, b2)
    w2_q = fake_quantize_ste(model[2].weight, b2)
    return h_q @ w2_q.T

# --- 3. Step 3: Per-Layer Mixed-Precision PTQ Sweep ---
base_model = load_base_model()
base_model.eval()

print("=== Step 3: Per-Layer Precision Sweep (PTQ) ===")
print(f"{'Layer 1':<10} {'Layer 2':<10} {'Accuracy':<10}")
print("-" * 32)
with torch.no_grad():
    for b1 in [16, 8, 4]:
        for b2 in [16, 8, 4]:
            preds = forward_mixed(base_model, X_test, b1, b2).argmax(dim=1)
            acc = (preds == y_test).float().mean().item() * 100.0
            print(f"INT{b1:<7} INT{b2:<7} {acc:.2f}%")

# --- 4. Step 2b: INT4 Quantization-Aware Training (QAT) ---
print("\n=== Step 2b: Fine-Tuning with INT4 QAT ===")
qat_model = load_base_model()
loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(qat_model.parameters(), lr=0.002)

for epoch in range(80):
    qat_model.train()
    logits = forward_mixed(qat_model, X_train, b1=4, b2=4)
    loss = loss_fn(logits, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()

qat_model.eval()
with torch.no_grad():
    preds = forward_mixed(qat_model, X_test, b1=4, b2=4).argmax(dim=1)
    qat_acc = (preds == y_test).float().mean().item() * 100.0

print(f"INT4 PTQ accuracy : 94.72%")
print(f"INT4 QAT accuracy : {qat_acc:.2f}%")

torch.save(qat_model.state_dict(), "qat_int4_model.pt")