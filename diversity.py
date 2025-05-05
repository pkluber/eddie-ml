from pathlib import Path

from ase import Atoms
from ase.io import read as ase_read

from collections import Counter, OrderedDict

import matplotlib.pyplot as plt
import numpy as np

def get_dataset_files(data_folder: Path):
    return [f for f in data_folder.iterdir() if f.is_file() and f.name.endswith('.xyz')]

def get_dataset_atoms(data_folder: Path):
    files = get_dataset_files(data_folder)
    atom_systems = []
    for file in files:
        atom_systems.append(ase_read(file))

    return atom_systems

def count_dataset_elements(data_folder: Path, del_h: bool = False):
    atoms_dataset = get_dataset_atoms(data_folder)
    counts = Counter()
    for atoms in atoms_dataset:
        counts.update(list(atoms.symbols))

    if del_h:
        del counts['H']

    total = sum(counts.values())
    percentages = {element: count / total * 100 for element, count in counts.items()}

    # Sort the dictionary by percentages in descending order
    sorted_percentages = OrderedDict(sorted(percentages.items(), key=lambda item: item[1], reverse=True))

    print(f'Element composition for dataset {data_folder.name}:')
    for element, percent in sorted_percentages.items():
        print(f'{element}: {percent:.2f}%')


if __name__ == '__main__':
    count_dataset_elements(Path('data/bcurves'))
    count_dataset_elements(Path('data/bcurves/extraILs'))
