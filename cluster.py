import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

import h5py

from utils import get_dataset, get_charge_from_position

# Open the HDF5 file
with h5py.File('output.hdf5', 'r') as f:
    X = f['value'][:]
    systems = f['system'][:]
    species = f['species'][:]
    positions = f['position'][:]

systems = [system.decode('ascii') for system in systems]
species = [element.decode('ascii') for element in species]
positions = [np.array(pos) for pos in positions]

# Parse energies.dat
energies = {}
with open('energies.dat') as fd:
    lines = fd.readlines()
    for line in lines:
        xyz, ie = line.split(' ')
        system_name = xyz[:-4]
        ie = float(ie)
    
        energies[system_name] = ie

y = np.array([energies[system] for system in systems])

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_embedded = tsne.fit_transform(X)

def plot(y: np.ndarray, label: str,  file_label: str = ''):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap="jet", alpha=0.5, s=10)
    plt.colorbar(scatter, label=label)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('t-SNE Visualization of Features')
    plt.savefig(f'tsne{file_label}.png', dpi=300, bbox_inches='tight')

plot(y * 627.509, 'Interaction Energy (kcal/mol)')

# Now do a second plot using distance as a proxy for % electrostatic
y = []
for system in systems:
    dist_str = system.split('-d')[1]
    dist_str = dist_str.replace('_', '.')
    y.append(float(dist_str))

y = np.array(y)

plot(y / 0.529, 'Distance (A)', file_label='_d')

# Now do a third plot visualizing the dataset  
plt.figure(figsize=(8, 6))
datasets = {}
for x in range(len(X)):
    dataset = get_dataset(systems[x])
    if dataset not in datasets:
        datasets[dataset] = [X_embedded[x]]
    else:
        datasets[dataset].append(X_embedded[x])

colors = {'S66': 'red', 'SSI': 'green', 'IL174': 'blue', 'extraILs': 'purple'}
for dataset, color in colors.items():
    Xs = np.array(datasets[dataset])
    plt.scatter(Xs[:, 0], Xs[:, 1], color=color, label=dataset, s=5)

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('t-SNE Visualization of Features')
plt.legend()
plt.savefig('tsne_datasets.png', dpi=300, bbox_inches='tight')

# Now do a fourth plot visualizing the species
plt.figure(figsize=(8, 6))
elements = {}
for x in range(len(X)):
    element = species[x]
    if element not in elements:
        elements[element] = [X_embedded[x]]
    else:
        elements[element].append(X_embedded[x])

colors = {'O': 'red', 'N': 'blue', 'H': 'purple', 'C': 'gray'} 
for element, color in colors.items():
    Xs = np.array(elements[element])
    plt.scatter(Xs[:, 0], Xs[:, 1], color=color, label=element, s=5)

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('t-SNE Visualization of Features')
plt.legend()
plt.savefig('tsne_elements.png', dpi=300, bbox_inches='tight')

# Now do a fifth plot visualizing the charge of the monomer
plt.figure(figsize=(8, 6))
charges = {}
for x in range(len(X)):
    position = positions[x]
    charge = get_charge_from_position(position)
    if charge not in charges:
        charges[charge] = [X_embedded[x]]
    else:
        charges[charge].append(X_embedded[x])

colors = {-1: 'black', 0: 'blue', 1: 'red'}
for charge, color in colors.items():
    Xs = np.array(charges[charge])
    plt.scatter(Xs[:, 0], Xs[:, 1], color=color, label=str(charge), s=5)

plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('t-SNE Visualization of Features')
plt.legend()
plt.savefig('tsne_charge.png', dpi=300, bbox_inches='tight')
