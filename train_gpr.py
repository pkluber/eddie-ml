import h5py
from pathlib import Path
import re
import numpy as np

# Open output hdf5
with h5py.File('output-neutral_pca.hdf5', 'r') as fd:
    X = fd['value'][:]
    systems = fd['system'][:]

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

Natoms = {}
AtomTypes = {}

pattern = r'\s*-?[0-9]*\.[0-9]*'
pattern = pattern + pattern + pattern

data_dir = Path('data/bcurves')
for file in data_dir.rglob('*'):
    if file.is_file() and file.suffix == '.xyz':
        sys_name = file.name[:-4]
        if sys_name not in energies:
            continue

        with open(file, 'r') as fd:
            Natoms[sys_name] = int(fd.readline())
            
            atom_list = []
            for line in fd:
                if re.search('[A-Z][a-z]?' + pattern, line):
                    line = line.strip()
                    line = line.split()
                    atom_list.append(line[0])

            AtomTypes[sys_name] = atom_list

# Convert for model.fit call
y = np.array([energies[system] for system in systems])
natoms = np.array([Natoms[system] for system in systems])
atomtypes = np.array([AtomTypes[system] for system in systems], dtype=object)

X = X.reshape(y.shape[0], -1)
y = y.reshape(-1, 1)

print(X.shape, y.shape, natoms.shape, atomtypes.shape)

from kernels.crossvalidate import elemental_kernel_CV

kp_grid = {'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
           'kernel_func': ['cosine', 'euclidean'],
           'lambda': [1e-6, 1e-5, 1e-4]}

model = elemental_kernel_CV(krr_param_grid=kp_grid)
model.fit(X, y, natoms, atomtypes, show_plot=False)
