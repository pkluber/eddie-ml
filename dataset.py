from pathlib import Path 
import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from sklearn.preprocessing import StandardScaler, RobustScaler
from joblib import dump, load
from typing import Tuple

from utils import get_dataset, get_datasets_list
from random import shuffle

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
    
    def scale_and_save_scalers(self, name: str = 'train') -> Tuple[StandardScaler, RobustScaler]:
        scaler_x = StandardScaler()
        X_shape = self.X.shape
        self.X = scaler_x.fit_transform(self.X.reshape(-1, X_shape[-1]))
        self.X = self.X.reshape(*X_shape)
        self.X = torch.from_numpy(self.X)
        dump(scaler_x, f'scaler_{name}.joblib')

        scaler_y = RobustScaler()
        Y_shape = self.Y.shape
        self.Y = scaler_y.fit_transform(self.Y.reshape(-1,1))
        self.Y = self.Y.reshape(*Y_shape)
        self.Y = torch.from_numpy(self.Y)
        dump(scaler_y, f'scaler_y_{name}.joblib')
        
        return scaler_x, scaler_y

    def apply_scalers(self, scaler_x: StandardScaler, scaler_y: RobustScaler):
        X_shape = self.X.shape
        self.X = scaler_x.transform(self.X.reshape(-1, X_shape[-1]))
        self.X = self.X.reshape(*X_shape)
        self.X = torch.from_numpy(self.X)

        Y_shape = self.Y.shape
        self.Y = scaler_y.transform(self.Y.reshape(-1,1))
        self.Y = self.Y.reshape(*Y_shape)
        self.Y = torch.from_numpy(self.Y)
    
    def load_and_apply_scalers(self, name: str = 'train') -> Tuple[StandardScaler, RobustScaler]:
        scaler_x = load(f'scaler_{name}.joblib')
        scaler_y = load(f'scaler_y_{name}.joblib')
        self.apply_scalers(scaler_x, scaler_y)
        return scaler_x, scaler_y
 
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return self.get(index)

class UEDDIESubset(UEDDIEDataset):
    def __init__(self, dataset: UEDDIEDataset, subset_ratios: dict[str, float]):
        self.dataset = dataset
        self.subset_ratios = subset_ratios
        
        # Count data from each dataset
        counts = {}
        for x in range(len(dataset)):
            _, _, _, _, name = dataset.get(x, return_name = True) 
            subset = get_dataset(name)
            if subset in subset_ratios.keys():
                if subset not in counts:
                    counts[subset] = 1
                else:
                    counts[subset] += 1
        
        # Create counts to pull from
        for subset in counts:
            if subset not in subset_ratios:
                del counts[subset]
                continue
            
            counts[subset] = int(counts[subset] * subset_ratios[subset])

        # Randomly access underlying dataset to form data
        self.X = []
        self.E = []
        self.C = []
        self.Y = []
        self.systems = []

        indices = list(range(len(dataset)))
        shuffle(indices)
        for x in indices:
            X, E, C, Y, system = dataset.get(x, return_name = True)
            subset = get_dataset(system)
            if subset not in counts:
                continue
            
            self.X.append(X)
            self.E.append(E)
            self.C.append(C)
            self.Y.append(Y)
            self.systems.append(system)

            new_count = counts[subset] - 1
            if new_count == 0:
                del counts[subset]
                continue

            counts[subset] = new_count
        
        self.X = torch.stack(self.X)
        self.E = torch.stack(self.E)
        self.C = torch.stack(self.C)
        self.Y = torch.stack(self.Y)
        

def get_dataloader(batch_size: int = 16, shuffle: bool = True):
    dataset = UEDDIEDataset()
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_train_validation_test_datasets(train_ratios: dict[str, float] | None = None):
    if train_ratios is None:
        train_ratios = {'IL174': 0.9, 'extraILs': 0.9, 'S66': 0.8, 'SSI': 0.8}

    base_dataset = UEDDIEDataset()
    
    train_dataset = UEDDIESubset(base_dataset, train_ratios)
    
    subsets = get_datasets_list()
    test_and_val_ratios = {}
    for subset in subsets:
        if subset not in train_ratios:
            test_and_val_ratios[subset] = 0.5
        else:
            test_and_val_ratios[subset] = (1.0 - train_ratios[subset]) / 2

    validation_dataset = UEDDIESubset(base_dataset, test_and_val_ratios)
    test_dataset = UEDDIESubset(base_dataset, test_and_val_ratios)

    return train_dataset, validation_dataset, test_dataset


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
