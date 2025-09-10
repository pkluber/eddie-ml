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

# Setup devices
devices = []
num_gpus = torch.cuda.device_count()
if num_gpus == 0:
    devices.append(torch.device('cpu'))
else:
    for x in range(num_gpus):
        print(f'Found GPU {x}: {torch.cuda.get_device_name(x)}')
        devices.append(torch.device(f'cuda:{x}'))

device = devices[0]

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
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=4, shuffle=True, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

# Initialize model
x_sample, _, _, _ = next(iter(train_dataloader))
d_model = x_sample.shape[-1]
model = UEDDIENetwork(d_model, num_heads=4, d_ff=128, depth_e=5, depth_c=5, multi_gpu=(num_gpus==4))
if len(devices) == 1:
    model.to(device) 

finetuner = UEDDIEFinetuner(device, x_sample.shape)
finetuner.to(device)

# Loss and stuff
loss_function = nn.MSELoss()
optimizer = optim.AdamW(list(model.parameters()) + list(finetuner.parameters()), lr=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=60)

print(f'Beginning training using primarily device={device}!', flush=True)

train_losses = []

n_epoch = 1000
for epoch in range(n_epoch):
    # Set LR to 1e-4 for the finetuner initially 
    if epoch == 1000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4
    
    # Training...
    model.train()

    train_loss = 0
    for X, E, C, Y in train_dataloader:
        X, E, C, Y = X.to(device, non_blocking=True), E.to(device, non_blocking=True), C.to(device, non_blocking=True), Y.to(device, non_blocking=True)
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
            X, E, C, Y = X.to(device, non_blocking=True), E.to(device, non_blocking=True), C.to(device, non_blocking=True), Y.to(device, non_blocking=True)

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

test_loss = 0
with torch.no_grad():
    for X, E, C, Y in test_dataloader:
        X, E, C, Y = X.to(device), E.to(device), C.to(device), Y.to(device)
        Y_pred = model(X, E, C) + finetuner(X, E, C)
        loss = loss_function(Y_pred, Y)
        test_loss += loss.item()

print(f'Test average loss: {test_loss / len(test_dataset)}')


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

ies = []
ies_pred = []
with torch.no_grad():
    for x, e, c, y, name in [test_dataset.get(x, return_name=True) for x in range(len(test_dataset))]:
        print(f'Testing {name}...')
        x, e, c = x.to(device), e.to(device), c.to(device)
        x = x.unsqueeze(0)
        e = e.unsqueeze(0)
        c = c.unsqueeze(0)

        y_pred = model(x, e, c).cpu()
        y_pred = np.array([y_pred.item()])
        y_pred = y_pred.reshape(1, 1)
        ie_pred = scaler_y.inverse_transform(y_pred)[0, 0] * 627.509  # kcal/mol

        y = y.cpu().item()
        y = np.array([y]).reshape(1, 1)
        ie = scaler_y.inverse_transform(y)[0, 0] * 627.509

        print(f'Predicted IE (kcal/mol): {ie_pred:.1f}')
        print(f'Actual IE    (kcal/mol): {ie:.1f}')

        ies.append(ie)
        ies_pred.append(ie_pred)

ies = np.array(ies)
ies_pred = np.array(ies_pred)

# Compute metrics
mse = mean_squared_error(ies, ies_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(ies, ies_pred)
r2 = r2_score(ies, ies_pred)

print(f'MSE:  {mse:.1f}')
print(f'RMSE: {rmse:.1f}')
print(f'MAE:  {mae:.1f}')
print(f'R^2:  {r2:.2f}')

# Best fit line
reg = LinearRegression().fit(ies_pred.reshape(-1, 1), ies)
slope = reg.coef_[0]
intercept = reg.intercept_

# Visualize test results
plt.scatter(ies_pred, ies, alpha=0.8)
line = np.linspace(min(ies_pred), max(ies_pred), 100)
plt.plot(line, line, 'r--', label='Ideal (y=x)')
plt.plot(line, slope*line + intercept, 'g-', label='Best-fit line')
plt.xlabel('Predicted $\\Delta E^{\\text{INT}}$ (kcal/mol)')
plt.ylabel('Actual $\\Delta E^{\\text{INT}}$ (kcal/mol)')
#plt.title('Predicted vs. Actual Interaction Energy')
plt.legend()

plt.text(
    0.05, 0.95, f'$R^2 = {r2:.2f}$',
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='square,pad=0.3')
)

plt.tight_layout()
plt.savefig('test_results.png', dpi=300)
