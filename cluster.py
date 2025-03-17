import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

import h5py

# Open the HDF5 file
with h5py.File('output.hdf5', 'r') as f:
    X = f['value'][:]
    systems = f['system'][:]

systems = [system.decode('ascii') for system in systems]

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
    plt.savefig(f'tsne{file_label}.png')

plot(y * 627.509, 'Interaction Energy (kcal/mol)')

# Now do a second plot using distance as a proxy for % electrostatic
y = []
for system in systems:
    print(system)
    dist_str = system.split('-d')[1]
    dist_str = dist_str.replace('_', '.')
    y.append(float(dist_str))

y = np.array(y)

plot(y / 0.529, 'Distance (A)', file_label='_d')
