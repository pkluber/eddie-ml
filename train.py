import torch 
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dataset
from model import UEDDIENetwork
from pathlib import Path

# Needed for 64-bit precision
torch.set_default_dtype(torch.float64)

# Load dataloader
dataloader = dataset.get_dataloader()

# Initialize model
x_sample, _, _ = next(iter(dataloader))
model = UEDDIENetwork(x_sample.shape)

# Loss and stuff
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-7)

print('Beginning training!')
n_epoch = 200
for epoch in range(n_epoch):
    for X, E, Y in dataloader:
        optimizer.zero_grad()
        
        Y_pred = model(X, E)
        loss = loss_function(Y_pred, Y)
        loss = torch.autograd.Variable(loss, requires_grad = True)
        
        loss.backward()
        optimizer.step()

    if epoch % 5 == 0:
        print(f'Epoch {epoch}, loss: {loss.item()}')

torch.save(model, 'model.pt')
