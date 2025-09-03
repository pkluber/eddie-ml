from model import UEDDIENetwork
from dataset import UEDDIEDataset
import torch 
import torch.nn as nn
import random

model = torch.load('model.pt')
model.eval()

criterion = nn.MSELoss()
total_loss = 0

dataset = UEDDIEDataset()

with torch.no_grad():
    test_id = random.randint(0, len(dataset) - 1)
    x, e, y, name = dataset.get(test_id, return_name = True)
    print(f'Testing system {name}!')

    x = x.unsqueeze(dim=0)
    e = e.unsqueeze(dim=0)
    y = y.unsqueeze(dim=0)

    y_pred = model(x, e)
    print(f'Predicted IE: {y_pred.item()}\nActual IE: {y.item()}')


