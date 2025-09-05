from pathlib import Path 
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py

def pad(nested_list, max_size, default):
    results = []
    for l in nested_list:
        if len(l) < max_size:
            l += [default] * (max_size - len(l))

        results.append(l)

    return results

class UEDDIEDataset(Dataset):
    def __init__(self, hdf5_path: Path = Path('output-pca.hdf5'), energies_path: Path = Path('energies.dat')):
        with h5py.File(hdf5_path, 'r') as f:
            X = f['value'][:]
            S = [system.decode('ascii') for system in f['system'][:]]
            E = [element.decode('ascii') for element in f['species'][:]]
            C = [int(round(charge)) for charge in f['charge'][:]]

        self.systems = []

        self.energies = {}
        with open(energies_path) as fd:
            lines = fd.readlines()
            for line in lines:
                xyz, ie = line.split(' ')
                system_name = xyz[:-4]
                ie = float(ie)
                
                self.systems.append(system_name)
                self.energies[system_name] = ie
        
        self.atoms = {}
        for xi in range(len(X)):
            x = X[xi]
            s = S[xi]
            e = E[xi]
            c = C[xi]
            elem = (e, c, x)
            if s not in self.atoms:
                self.atoms[s] = [elem]
            else:
                self.atoms[s].append(elem)
        
        X_list = [[x for _, _, x in self.atoms[s]] for s in self.systems]
        E_list = [[e for e, _, _ in self.atoms[s]] for s in self.systems]
        C_list = [[c for _, c, _ in self.atoms[s]] for s in self.systems]
        
        # Convert E_list to numerical form
        conversion_key = {'H': 0, 'C': 1, 'N': 2, 'O': 3}
        E_list = [[conversion_key[e] for e in e_list] for e_list in E_list]

        # Pad X_list, C_list, and E_list  
        max_size = 0
        for x in X_list:
            if len(x) > max_size:
                max_size = len(x)

        X_list = pad(X_list, max_size, [0.0] * len(X_list[0][0]))
        E_list = pad(E_list, max_size, -1)
        C_list = pad(C_list, max_size, 0)

        # Finally convert to torch tensors
        self.X = torch.tensor(np.array(X_list))
        self.E = torch.tensor(np.array(E_list))
        self.C = torch.tensor(np.array(C_list))

        # Don't forget about energies
        self.Y = torch.tensor(np.array(list(self.energies.values())))

        print(self.X.shape)
        print(self.Y.shape)


    def get(self, index, return_name: bool = False):
        if return_name:
            return self.X[index, ...], self.E[index, ...], self.C[index, ...], self.Y[index, ...], self.systems[index]

        return self.X[index, ...], self.E[index, ...], self.C[index, ...], self.Y[index, ...]
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        return self.get(index)

def get_dataloader(batch_size: int = 16, shuffle: bool = True):
    dataset = UEDDIEDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == '__main__':
    dataloader = get_dataloader()
    x_sample, e_sample, c_sample, y_sample = next(iter(dataloader))
    
    print(f'x shape: {x_sample.shape}')
    print(f'e shape: {e_sample.shape}')
    print(f'c shape: {c_sample.shape}')
    print(f'y shape: {y_sample.shape}')

    print(f'x dtype: {x_sample.dtype}')
    print(f'e dtype: {e_sample.dtype}')
    print(f'c dtype: {c_sample.dtype}')
    print(f'y dtype: {y_sample.dtype}')



