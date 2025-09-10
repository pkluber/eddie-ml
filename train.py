import torch 
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import numpy as np
import dataset
from model import UEDDIEMoE, UEDDIENetwork, UEDDIEFinetuner
from pathlib import Path

# Needed for 64-bit precision
torch.set_default_dtype(torch.float64)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load datasets
train_dataset, validation_dataset, test_dataset = dataset.get_train_validation_test_datasets()
total_dataset = len(train_dataset) + len(validation_dataset) + len(test_dataset)
train_split_percent = int(100 * len(train_dataset) / total_dataset)
validation_split_percent = int(100 * len(validation_dataset) / total_dataset)
print(f'Using {train_split_percent}/{validation_split_percent}/{100-train_split_percent-validation_split_percent} train/validation/test split')

# Scale data
scaler_x, scaler_y = train_dataset.scale_and_save_scalers()
validation_dataset.apply_scalers(scaler_x, scaler_y)
test_dataset.apply_scalers(scaler_x, scaler_y)

# Initialize dataloaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Initialize model
x_sample, _, _, _ = next(iter(train_dataloader))
d_model = x_sample.shape[-1]
model = UEDDIENetwork(d_model, num_heads=4, d_ff=128, depth_e=10, depth_c=10)
model.to(device)

finetuner = UEDDIEFinetuner(device, x_sample.shape)
model.to(device)

# Loss and stuff
loss_function = nn.MSELoss()
optimizer = optim.AdamW(list(model.parameters()) + list(finetuner.parameters()), lr=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=60)

print(f'Beginning training using device={device}!', flush=True)

train_losses = []

n_epoch = 2000
for epoch in range(n_epoch):
    # Set LR to 1e-4 for the finetuner initially 
    if epoch == 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4
    
    # Training...
    model.train()

    train_loss = 0
    for X, E, C, Y in train_dataloader:
        X, E, C, Y = X.to(device), E.to(device), C.to(device), Y.to(device)
        optimizer.zero_grad()
        
        if epoch < n_epoch // 2: # For 1-1000 epochs train base model
            Y_pred = model(X, E, C)
        else: # For 1000-2000 epochs train finetuner 
            Y_pred = model(X, E, C).detach() + finetuner(X, E, C)
        loss = loss_function(Y_pred, Y) 
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_losses.append(train_loss / len(train_dataset))
    
    # Validation...
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, E, C, Y in validation_dataloader:
            X, E, C, Y = X.to(device), E.to(device), C.to(device), Y.to(device)

            if epoch < n_epoch // 2:
                Y_pred = model(X, E, C)
            else:
                Y_pred = model(X, E, C) + finetuner(X, E, C)

            loss = loss_function(Y_pred, Y) 
            val_loss += loss.item()

    val_loss /= len(validation_dataloader)
    
    # Step plateau scheduler
    scheduler.step(val_loss)
    
    # Output and save 
    if epoch % 5 == 0:
        print(f'Epoch {epoch}, train loss: {train_losses[-1]}, val loss: {val_loss}, LR: {optimizer.param_groups[0]["lr"]}', flush=True)
        np.save('losses.npy', np.array(train_losses))
        torch.save(model, 'model.pt')
        torch.save(finetuner, 'finetuner.pt')

#TODO more for testing
test_loss = 0
with torch.no_grad():
    for X, E, C, Y in test_dataloader:
        X, E, C, Y = X.to(device), E.to(device), C.to(device), Y.to(device)
        Y_pred = model(X, E, C) + finetuner(X, E, C)
        loss = loss_function(Y_pred, Y)
        test_loss += loss.item()

print(f'Test average loss: {test_loss / len(test_dataset)}')
    

