import numpy as np
from pathlib import Path
from typing import Tuple

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

def get_datasets_list():
    return ['S66', 'SSI', 'IL174', 'extraILs']

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


