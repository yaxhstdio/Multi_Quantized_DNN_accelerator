import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(42)

# --- 1. Load data ---
digits = load_digits()
X = digits.data / 16.0
y = digits.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

# --- 2. Symmetric PTQ helper ---
def fake_quantize(tensor, bits):
    if bits is None:
        return tensor
    q_max = (1 << (bits - 1)) - 1
    max_val = tensor.abs().max()
    if max_val == 0:
        return tensor
    scale = max_val / q_max
    return torch.clamp(torch.round(tensor / scale), -q_max, q_max) * scale

# --- 3. Flexible MLP class (bias=False to match RTL) ---
class FlexibleMLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1], bias=False) for i in range(len(dims) - 1)
        ])

    def forward(self, x, bit_list=None):
        for i, layer in enumerate(self.layers):
            b = None if bit_list is None else bit_list[i]
            x_q = fake_quantize(x, b)
            w_q = fake_quantize(layer.weight, b)
            x = x_q @ w_q.T
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x

def train_model(dims, epochs=300, lr=0.005):
    model = FlexibleMLP(dims)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X_train), y_train)
        loss.backward()
        opt.step()
    model.eval()
    return model

def eval_model(model, bit_list=None):
    with torch.no_grad():
        preds = model(X_test, bit_list).argmax(dim=1)
        return (preds == y_test).float().mean().item() * 100.0

# --- 4. Train 2-Layer vs 5-Layer MLPs ---
print("Training 2-layer and 5-layer MLPs...")
mlp_2 = train_model([64, 16, 10], epochs=250, lr=0.01)
mlp_5 = train_model([64, 64, 64, 64, 32, 10], epochs=350, lr=0.005)

# --- 5. Evaluate across quantization schemes ---
acc_2_fp32  = eval_model(mlp_2, None)
acc_2_int4  = eval_model(mlp_2, [4, 4])
acc_2_mixed = eval_model(mlp_2, [8, 8]) # 2-layer has no middle layers

acc_5_fp32  = eval_model(mlp_5, None)
acc_5_int4  = eval_model(mlp_5, [4, 4, 4, 4, 4])
acc_5_mixed = eval_model(mlp_5, [8, 4, 4, 4, 8]) # First & Last = INT8, Middle = INT4

print("\n=== Depth vs. Quantization Sensitivity (PTQ) ===")
print(f"{'Model':<22} {'FP32':<10} {'Uniform INT4':<15} {'Mixed (8-4-...-8)':<18}")
print("-" * 68)
print(f"{'2-Layer (64-16-10)':<22} {acc_2_fp32:<10.2f} {acc_2_int4:<15.2f} {'N/A (no mid)':<18}")
print(f"{'5-Layer (Deep MLP)':<22} {acc_5_fp32:<10.2f} {acc_5_int4:<15.2f} {acc_5_mixed:<18.2f}")

# --- 6. Plot for presentation slides ---
labels = ["FP32 Baseline", "Uniform INT4\n(All Layers)", "Mixed Precision\n(First/Last INT8, Mid INT4)"]
scores_2 = [acc_2_fp32, acc_2_int4, np.nan]
scores_5 = [acc_5_fp32, acc_5_int4, acc_5_mixed]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
rects1 = ax.bar(x[:2] - width/2, scores_2[:2], width, label="2-Layer MLP (Shallow)", color="#3498db")
rects2 = ax.bar(x + width/2, scores_5, width, label="5-Layer MLP (Deep)", color="#e67e22")

ax.set_ylabel("Test Accuracy (%)", fontsize=11)
ax.set_title("Why Runtime Mixed-Precision Matters as Network Depth Grows", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylim(max(0, min(acc_5_int4, acc_2_int4) - 15), 102)
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5)

for rect in list(rects1) + list(rects2):
    h = rect.get_height()
    if not np.isnan(h):
        ax.annotate(f"{h:.1f}%", xy=(rect.get_x() + rect.get_width()/2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontweight="bold")

plt.tight_layout()
plt.savefig("depth_vs_quantization.png", dpi=200)
print("\nSaved chart -> depth_vs_quantization.png")
plt.show()