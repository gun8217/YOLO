from sklearn.datasets import make_blobs
from torch import FloatTensor, LongTensor  # LongTensor : torch.int64
from torch.utils.data import TensorDataset, DataLoader

import torch
import torch.nn as nn
from torch.optim import SGD

from model04 import MLP

n_samples = 100
n_classes = 4
batch_size = 8

X, y = make_blobs(n_samples=n_samples, centers=n_classes,
                  n_features=2, cluster_std=0.7, random_state=1)

dataset = TensorDataset(FloatTensor(X), LongTensor(y))
dataloader = DataLoader(dataset, batch_size=batch_size)

BATCH_SIZE = 8
LR = 0.01
EPOCHS = 100

if torch.cuda.is_available(): DEVICE = 'cuda'
elif torch.backends.mps.is_available(): DEVICE = 'mps'
else: DEVICE = 'cpu'

model = MLP().to(DEVICE)
loss_function = nn.CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=LR)

losses = []
for epoch in range(EPOCHS):
    epoch_loss = 0.
    for X_, y_ in dataloader:
        X_, y_ = X_.to(DEVICE), y_.to(DEVICE)
        
        pred = model(X_)
        loss = loss_function(pred, y_)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * len(X_)
        
    epoch_loss /= n_samples
    losses.append(epoch_loss)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Train Loss = {loss.item():.4f}, Val Loss = {epoch_loss.item():.4f}")
    
    # print(f"Epoch: {epoch + 1}")
    # print(f"Loss: {epoch_loss:.4f}")


import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(losses)

ax.set_xlabel("Epoch", fontsize=15)
ax.set_ylabel("Cross Entropy Loss", fontsize=15)
ax.tick_params(labelsize=10)

fig.tight_layout()
plt.show()