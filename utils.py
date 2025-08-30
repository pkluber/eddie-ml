import numpy as np
from pathlib import Path
from typing import Tuple

POS_CHARGED_AMINOS = ['ARG', 'LYS']
NEG_CHARGED_AMINOS = ['ASP', 'GLU']

def get_amino_charge(amino: str) -> int:
    if amino in POS_CHARGED_AMINOS:
        return 1
    elif amino in NEG_CHARGED_AMINOS:
        return -1
    else:
        return 0

def get_charges(filename: str) -> Tuple[int, int, int]:
    if filename.startswith('S66'):
        return 0, 0, 0
    elif filename.startswith('SSI'):
        split = filename.split('-')
        aa1 = split[1][3:]
        aa2 = split[2][3:]
        charge1 = get_amino_charge(aa1)
        charge2 = get_amino_charge(aa2)
        if charge1 == charge2 or charge1 + charge2 != 0: 
            return 0, 0, 0
        else:
            return 0, charge1, charge2
    elif filename.startswith('C_'): # IL174
        return 0, 1, -1
    elif filename.startswith('C'): # extraILs
        if filename.startswith('C0491_A0090') or filename.startswith('C2004_A0073'):
            return 0, -1, 1
        else:
            return 0, 1, -1

def get_dataset(filename: str) -> str:
    if filename.startswith('S66'):
        return 'S66'
    elif filename.startswith('SSI'):
        return 'SSI'
    elif filename.startswith('C_'):
        return 'IL174'
    elif filename.startswith('C'):
        return 'extraILs'
    else:
        return 'Unknown'

def geom_from_xyz_dimer(filename: str, charges: Tuple[int, int, int]) -> Tuple[str, str, str] | None:
    with open(filename) as fd:
        lines = fd.readlines() # note preserves \n characters 
        try:
            num_atoms_m1 = int(lines[0])
            num_atoms_m2 = int(lines[num_atoms_m1+2])
            
            m1_start = 2
            geometry_m1 = "".join(lines[m1_start:m1_start+num_atoms_m1])
            m2_start = m1_start + num_atoms_m1 + 2
            geometry_m2 = "".join(lines[m2_start:m2_start+num_atoms_m2])

            return f'{charges[0]} 1\n{geometry_m1+geometry_m2}', \
                    f'{charges[1]} 1\n{geometry_m1}', \
                    f'{charges[2]} 1\n{geometry_m2}'
        except ValueError:
            print(f'Error parsing xyz file {filename}')
            return None

GHOSTS_FOOTER = 'no_com\nno_reorient\n'
def geom_from_xyz_dimer_ghosts(filename: str, charges: Tuple[int, int, int]) -> Tuple[str, str, str, str] | None:
    with open(filename) as fd:
        lines = fd.readlines() # note preserves \n characters 
        try:
            num_atoms_m1 = int(lines[0])
            num_atoms_m2 = int(lines[num_atoms_m1+2])
            
            m1_start = 2
            geometry_m1 = "".join(lines[m1_start:m1_start+num_atoms_m1])
            m2_start = m1_start + num_atoms_m1 + 2
            geometry_m2 = "".join(lines[m2_start:m2_start+num_atoms_m2])

            ghosts_m1 = '@' + '@'.join(lines[m1_start:m1_start+num_atoms_m1])
            ghosts_m2 = '@' + '@'.join(lines[m2_start:m2_start+num_atoms_m2])

            return f'{charges[1]} 1\n{geometry_m1}--\n0 1\n{ghosts_m2}{GHOSTS_FOOTER}', \
                    f'{charges[2]} 1\n{geometry_m2}--\n0 1\n{ghosts_m1}{GHOSTS_FOOTER}', \
                    f'{charges[1]} 1\n{geometry_m1}{GHOSTS_FOOTER}', \
                    f'{charges[2]} 1\n{geometry_m2}{GHOSTS_FOOTER}'
        except ValueError:
            print(f'Error parsing xyz file {filename}')
            return None

def get_charge_from_position(system_name: str, position: np.ndarray) -> int | None:
    position = np.array(position)

    path = Path('data/bcurves')
    if system_name.startswith('C') and not system_name.startswith('C_'):
        path = path / 'extraILs'

    path = path / f'{system_name}.xyz'

    charges = get_charges(path)

    with open(path) as fd:
        lines = fd.readlines() # note preserves \n characters 
        try:
            num_atoms_m1 = int(lines[0])
            m1_start = 2
            for line in lines[m1_start:m1_start+num_atoms_m1]:
                split = line.strip().split()
                xyz = np.array([float(num) for num in split[1:]])
                if np.linalg.norm(xyz - position) < 1e-6:
                    return charges[1]

            return charges[2] 
        except ValueError:
            print(f'Error trying to parse xyz file {path}')
            return None

