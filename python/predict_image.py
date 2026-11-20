import argparse
import subprocess
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# --- Verified mode configs from your HIL runs ---
MODES = {
    "int16": ("float_model.pt",    "float_model.pt", 16, 16, 2, 17, 2, 17),
    "int8":  ("float_model.pt",    "float_model.pt",  8,  8, 1, 11, 1,  7),
    "int4":  ("qat_int4_model.pt", "qat_int4_model.pt", 4, 4, 0,  5, 2,  0),
    "mixed": ("qat_int4_model.pt", "float_model.pt",  4,  8, 0,  5, 1,  4),
}

parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True, help="Path to JPEG/PNG digit image")
parser.add_argument("--mode", choices=MODES.keys(), default="int8")
args = parser.parse_args()

w1_file, w2_file, b1, b2, m1, s1, m2, s2 = MODES[args.mode]

def load_w(path, idx):
    m = nn.Sequential(nn.Linear(64, 16, bias=False), nn.ReLU(), nn.Linear(16, 10, bias=False))
    m.load_state_dict(torch.load(path, weights_only=True))
    return m[idx].weight.detach().numpy()

def quantize_to_int(arr, bits):
    q_max = (1 << (bits - 1)) - 1
    max_v = np.max(np.abs(arr))
    return np.zeros_like(arr, dtype=np.int32) if max_v == 0 else np.clip(np.round(arr / (max_v / q_max)), -q_max, q_max).astype(np.int32)

# --- 1. Load & preprocess JPEG to 8x8 ---
orig_img = Image.open(args.image).convert("L")
arr_raw = np.array(orig_img, dtype=np.float32)

# Auto-invert if dark digit on white background (dataset is white digit on black)
if np.mean(arr_raw) > 127.0:
    proc_img = ImageOps.invert(orig_img)
else:
    proc_img = orig_img

img_8x8 = proc_img.resize((8, 8), Image.Resampling.BILINEAR)
x_norm = np.array(img_8x8, dtype=np.float32) / 255.0
x_int  = quantize_to_int(x_norm.flatten(), b1)

# --- 2. Export weights.hex & single_input.hex ---
w1_int = quantize_to_int(load_w(w1_file, 0), b1)
w2_int = quantize_to_int(load_w(w2_file, 2), b2)

bram_words = []
for p in range(4):
    for k in range(64):
        w = [w1_int[p*4 + pe, k] for pe in range(4)]
        bram_words.append(f"{w[3]&0xFFFF:04x}{w[2]&0xFFFF:04x}{w[1]&0xFFFF:04x}{w[0]&0xFFFF:04x}")
w2_pad = np.vstack([w2_int, np.zeros((2, 16), dtype=np.int32)])
for p in range(3):
    for k in range(16):
        w = [w2_pad[p*4 + pe, k] for pe in range(4)]
        bram_words.append(f"{w[3]&0xFFFF:04x}{w[2]&0xFFFF:04x}{w[1]&0xFFFF:04x}{w[0]&0xFFFF:04x}")
while len(bram_words) < 512:
    bram_words.append("0000000000000000")

with open("weights.hex", "w", newline="\n") as f:
    f.write("\n".join(bram_words) + "\n")
with open("single_input.hex", "w", newline="\n") as f:
    for val in x_int:
        f.write(f"{int(val) & 0xFFFF:04x}\n")

# --- 3. Copy to WSL & run Verilog RTL ---
wsl_cmd = (
    'cp "/mnt/c/Users/meyas/Desktop/study material/projects/MLCV + DNN accelerator/weights.hex" ~/EDI/ && '
    'cp "/mnt/c/Users/meyas/Desktop/study material/projects/MLCV + DNN accelerator/single_input.hex" ~/EDI/ && '
    f'cd ~/EDI && vvp sim_single.vvp +MODE_L1={m1} +SHIFT_L1={s1} +MODE_L2={m2} +SHIFT_L2={s2}'
)
res = subprocess.run(["wsl", "bash", "-c", wsl_cmd], capture_output=True, text=True, check=True)

logits, pred, cycles = None, None, None
for line in res.stdout.splitlines():
    if line.startswith("LOGITS:"):
        logits = [int(v) for v in line.split(":")[1].split(",")]
    elif line.startswith("PRED:"):
        pred = int(line.split(":")[1])
    elif line.startswith("CYCLES:"):
        cycles = int(line.split(":")[1])

print(f"RTL Prediction: {pred} | Mode: {args.mode.upper()} | Cycles: {cycles}")

# --- 4. Display JPEG, 8x8 Hardware Input, and RTL Output ---
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
fig.suptitle(f"Verilog RTL Prediction: {pred}   |   Mode: {args.mode.upper()}   |   {cycles} cycles", fontsize=14, fontweight="bold")

axes[0].imshow(orig_img, cmap="gray")
axes[0].set_title("Input JPEG")
axes[0].axis("off")

axes[1].imshow(x_int.reshape(8, 8), cmap="gray")
axes[1].set_title(f"8x8 Quantized ({b1}-bit)")
axes[1].axis("off")

colors = ["#2ecc71" if i == pred else "#3498db" for i in range(10)]
axes[2].bar(range(10), logits, color=colors)
axes[2].set_xticks(range(10))
axes[2].set_title("RTL Output Logits")
axes[2].set_xlabel("Digit Class")

plt.tight_layout()
plt.show()