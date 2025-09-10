import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Generate dissociation curve for a given system.')
parser.add_argument('--input', type=str, default='S66_01WaterWater', help='Input system name')

args = parser.parse_args()

energies = {}
with open('energies.dat') as fd:
    lines = fd.readlines()
    for line in lines:
        xyz, ie = line.split(' ')
        system_name = xyz[:-4]
        ie = float(ie) * 627.509
        
        energies[system_name] = ie

systems = []
for system in energies.keys():
    if system.startswith(args.input):
        systems.append(system)

def get_dist(system: str): 
    split = system.split('-')
    dist_raw = split[-1]
    num_raw = dist_raw.split('_')

    if len(num_raw) == 1:
        return int(num_raw)
    elif len(num_raw) == 2:
        return float(dist_raw[1:].replace('_', '.'))


xs = []
ys = []

for system in systems:
    xs.append(get_dist(system))
    ys.append(energies[system])

xs = np.array(xs)
ys = np.array(ys)

def fit(xs: np.ndarray, ys: np.ndarray, degree: int = 7):
    coefficients = np.polyfit(xs, ys, degree)
    polynomial = np.poly1d(coefficients)

    x_fit = np.linspace(min(xs), max(xs), 100)
    y_fit = polynomial(x_fit)

    return x_fit, y_fit

# Start plotting!
import matplotlib.pyplot as plt

def plot_and_fit(xs: np.ndarray, ys: np.ndarray, color: str, fit_color: str, degree: int = 7, should_fit: bool = True):
    if should_fit:
        x_fit, y_fit = fit(xs, ys, degree=degree)
    else:
        x_fit, y_fit = xs, ys
    plt.plot(x_fit, y_fit, color=fit_color)
    plt.scatter(xs, ys, color=color)

plot_and_fit(xs, ys, 'black', 'gray')

# Evaluate the model on each of the systems
import torch
from dataset import UEDDIEDataset
from model import UEDDIENetwork, UEDDIEFinetuner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load('model.pt', weights_only=False, map_location=device)
model.eval()
finetuner = torch.load('finetuner.pt', weights_only=False, map_location=device)
finetuner.eval()

dataset = UEDDIEDataset()
_, scaler_y = dataset.load_and_apply_scalers()

xs_model = []
ys_model = []
min_ie = 0
min_sys = ''
with torch.no_grad():
    for x, e, c, y, name in [dataset.get(x, return_name=True) for x in range(len(dataset))]:
        if name.startswith(args.input):
            x = x.unsqueeze(0)
            e = e.unsqueeze(0)
            c = c.unsqueeze(0)

            y_pred = model(x, e, c) + finetuner(x, e, c) 
            y_pred = np.array([y_pred.item()])
            y_pred = y_pred.reshape(1, 1)
            ie_model = scaler_y.inverse_transform(y_pred)[0, 0] * 627.509
            if ie_model < min_ie:
                min_ie = ie_model
                min_sys = name
            
            xs_model.append(get_dist(name))
            ys_model.append(ie_model)

print(f'Min IE system is {min_sys} with {min_ie} kcal/mol')

xs_model = np.array(xs_model)
ys_model = np.array(ys_model)

# Sort xs_model and ys_model
indices = xs_model.argsort()
xs_model = xs_model[indices]
ys_model = ys_model[indices]

plot_and_fit(xs_model, ys_model, 'blue', 'lightblue', should_fit=False)

plt.xlabel('Distance (A)')
plt.ylabel('Interaction Energy (kcal/mol)')
plt.title(f'Dissociation of {args.input}')
plt.savefig('dissociation.png')
