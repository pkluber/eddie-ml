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
dataloader = dataset.get_dataloader(batch_size=1)

# Initialize model
x_sample, _, _ = next(iter(dataloader))
model = UEDDIENetwork(x_sample.shape)

# Loss and stuff
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-6)

print(f'Beginning training using device={device}!', flush=True)
n_epoch = 1000
losses = []
for epoch in range(n_epoch):
    total_loss = 0
    for X, E, Y in dataloader:
        X, E, Y = X.to(device), E.to(device), Y.to(device)
        optimizer.zero_grad()
        
        Y_pred = model(X, E)
        loss = loss_function(Y_pred, Y) 
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    losses.append(total_loss)

    if epoch % 5 == 0:
        print(f'Epoch {epoch}, loss: {loss.item()}', flush=True)

torch.save(model, 'model.pt')
np.save('losses.npy', np.array(losses))
