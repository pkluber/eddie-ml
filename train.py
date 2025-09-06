import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dataset
from model import UEDDIENetwork
from pathlib import Path

# Needed for 64-bit precision
torch.set_default_dtype(torch.float64)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load dataloader
dataloader = dataset.get_dataloader(batch_size=4)

# Initialize model
x_sample, _, _, _ = next(iter(dataloader))
model = UEDDIENetwork(x_sample.shape)
model.to(device)

# Loss and stuff
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-6)

print(f'Beginning training using device={device}!', flush=True)
n_epoch = 2000
losses = []
for epoch in range(n_epoch):
    total_loss = 0
    for X, E, C, Y in dataloader:
        X, E, C, Y = X.to(device), E.to(device), C.to(device), Y.to(device)
        optimizer.zero_grad()
        
        Y_pred = model(X, E, C)
        loss = loss_function(Y_pred, Y) 
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    losses.append(total_loss)

    if epoch % 5 == 0:
        print(f'Epoch {epoch}, loss: {losses[-1]}', flush=True)

    if epoch % 20 == 0:
        torch.save(model, 'model.pt')
        np.save('losses.npy', np.array(losses))

