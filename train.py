import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import dataset
from model import UEDDIEMoE, UEDDIENetwork
from pathlib import Path

# Needed for 64-bit precision
torch.set_default_dtype(torch.float64)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataloaders
train_dataset, test_dataset = dataset.get_train_test_datasets()
total_dataset = len(train_dataset) + len(test_dataset)
split_percent = int(100 * len(train_dataset) / total_dataset)
print(f'Using {split_percent}/{100-split_percent} train/test split')

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize model
x_sample, _, _, _ = next(iter(train_dataloader))
model = UEDDIENetwork(x_sample.shape)
model.to(device)

# Loss and stuff
loss_function = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

print(f'Beginning training using device={device}!', flush=True)
n_epoch = 2000
losses = []
for epoch in range(n_epoch):
    total_loss = 0
    for X, E, C, Y in train_dataloader:
        X, E, C, Y = X.to(device), E.to(device), C.to(device), Y.to(device)
        optimizer.zero_grad()
        
        Y_pred = model(X, E, C)
        loss = loss_function(Y_pred, Y) 
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    losses.append(total_loss / len(train_dataset))

    if epoch % 5 == 0:
        print(f'Epoch {epoch}, average loss: {losses[-1]}', flush=True)

    if epoch % 20 == 0:
        torch.save(model, 'model.pt')
        np.save('losses.npy', np.array(losses))

test_loss = 0
with torch.no_grad():
    for X, E, C, Y in test_dataloader:
        X, E, C, Y = X.to(device), E.to(device), C.to(device), Y.to(device)
        Y_pred = model(X, E, C)
        loss = loss_function(Y_pred, Y)
        test_loss += loss.item()

print(f'Test average loss: {test_loss / len(test_dataset)}')
    

