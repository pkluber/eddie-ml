import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
train_dataset, validation_dataset, test_dataset = dataset.get_train_validation_test_datasets()
total_dataset = len(train_dataset) + len(validation_dataset) + len(test_dataset)
train_split_percent = int(100 * len(train_dataset) / total_dataset)
validation_split_percent = int(100 * len(validation_dataset) / total_dataset)
print(f'Using {train_split_percent}/{validation_split_percent}/{100-train_split_percent-validation_split_percent} train/validation/test split')

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize model
x_sample, _, _, _ = next(iter(train_dataloader))
model = UEDDIENetwork(x_sample.shape)
model.to(device)

# Loss and stuff
loss_function = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

print(f'Beginning training using device={device}!', flush=True)

train_losses = []

# Early stopping
early_stopping_patience = 40
best_val_loss = float('inf')
epochs_no_improve = 0

n_epoch = 2000
for epoch in range(n_epoch):
    model.train()

    train_loss = 0
    for X, E, C, Y in train_dataloader:
        X, E, C, Y = X.to(device), E.to(device), C.to(device), Y.to(device)
        optimizer.zero_grad()
        
        Y_pred = model(X, E, C)
        loss = loss_function(Y_pred, Y) 
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_losses.append(train_loss / len(train_dataset))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, E, C, Y in validation_dataloader:
            X, E, C, Y = X.to(device), E.to(device), C.to(device), Y.to(device)
            Y_pred = model(X, E, C)
            loss = loss_function(Y_pred, Y) 
            val_loss += loss.item()

    val_loss /= len(validation_dataloader)

    scheduler.step(val_loss)

    # Check for early stopping
    if val_loss < best_val_loss - 1e-7: # 1e-7 of tolerance
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model, 'model.pt')
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= early_stopping_patience:
        print(f'Early stopping at epoch {epoch}')
        break

    # Output
    if epoch % 5 == 0:
        print(f'Epoch {epoch}, train loss: {train_losses[-1]}, val loss: {val_loss}, LR: {optimizer.param_groups[0]["lr"]}', flush=True)
        np.save('losses.npy', np.array(train_losses))

#TODO more for testing
test_loss = 0
with torch.no_grad():
    for X, E, C, Y in test_dataloader:
        X, E, C, Y = X.to(device), E.to(device), C.to(device), Y.to(device)
        Y_pred = model(X, E, C)
        loss = loss_function(Y_pred, Y)
        test_loss += loss.item()

print(f'Test average loss: {test_loss / len(test_dataset)}')
    

