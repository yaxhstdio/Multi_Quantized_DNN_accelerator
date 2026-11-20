import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# --- 1. Load data: 1797 images, each 8x8 = 64 pixels, labels 0-9 ---
digits = load_digits()
X = digits.data / 16.0          # pixels are 0..16, scale to 0..1
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

X_train = torch.tensor(X_train, dtype=torch.float32)
X_test  = torch.tensor(X_test,  dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test  = torch.tensor(y_test,  dtype=torch.long)

# --- 2. Model: 64 -> 16 -> 10 ---
model = nn.Sequential(
    nn.Linear(64, 16, bias=False),
    nn.ReLU(),
    nn.Linear(16, 10, bias=False)
    # TODO: Linear layer from 64 inputs to 16 outputs
    # TODO: ReLU
    # TODO: Linear layer from 16 inputs to 10 outputs
)

loss_fn = nn.CrossEntropyLoss()
opt = torch.optim.Adam(model.parameters(), lr=0.01)

# --- 3. Training loop ---
for epoch in range(200):
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    opt.zero_grad()
    loss.backward()
    opt.step()
    # TODO: 1) get predictions from model on X_train
    #       2) compute loss against y_train
    #       3) opt.zero_grad(), loss.backward(), opt.step()
    if epoch % 20 == 0:
        print(f"epoch {epoch}  loss {loss.item():.4f}")

# --- 4. Evaluate ---
with torch.no_grad():
    preds = model(X_test).argmax(dim=1)
    acc = (preds == y_test).float().mean().item()
print(f"Float test accuracy: {acc*100:.2f}%")

# --- 5. Save for later quantization steps ---
torch.save(model.state_dict(), "float_model.pt")
np.save("X_test.npy", X_test.numpy())
np.save("y_test.npy", y_test.numpy())