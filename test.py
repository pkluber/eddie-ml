from model import UEDDIENetwork
from dataset import UEDDIEDataset
import torch 
import torch.nn as nn
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('model.pt', weights_only=False, map_location=device)
model.eval()

criterion = nn.MSELoss()
total_loss = 0

dataset = UEDDIEDataset()

with torch.no_grad():
    test_id = random.randint(0, len(dataset) - 1)
    x, e, c, y, name = dataset.get(test_id, return_name = True)
    print(f'Testing system {name}!')

    x, e, c, y = x.unsqueeze(0), e.unsqueeze(0), c.unsqueeze(0), y.unsqueeze(0)

    y_pred = model(x, e, c)

    y = y.item() * 627.509
    y_pred = y_pred.item() * 627.509

    print(f'Predicted interaction energy: {y_pred:.1f} kcal/mol\nActual interaction energy: {y:.1f} kcal/mol')


