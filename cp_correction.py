from pathlib import Path

from utils import get_charges, geom_from_xyz_dimer_ghosts

import psi4 

psi4.core.set_num_threads(48)
psi4.set_memory('80 GB')

energies_path = Path('energies.dat')

with open(energies_path) as fd:
    lines = fd.readlines()
    lines = [line.split(' ') for line in lines]
    energies = {system.strip(): float(value) for system, value in lines}

def cp_correct(cp_geom: str, geom: str):
    psi4.geometry(cp_geom)
    psi4.set_options({'basis': 'aug-cc-pvqz'})
    hf_cp_e = psi4.energy('scf')
    
    psi4.geometry(geom)
    hf_e = psi4.energy('scf')

    return hf_cp_e - hf_e


def cp_correction(mono1_cp_geom: str, mono2_cp_geom: str, mono1_geom: str, mono2_geom: str):
    return cp_correct(mono1_cp_geom, mono1_geom) + cp_correct(mono2_cp_geom, mono2_geom) 

# Load energies_cp.dat and avoid recomputing
cp_energies_path = Path('cp_energies.dat')
if cp_energies_path.is_file():
    with open(cp_energies_path) as fd:
        lines = fd.readlines()
        lines = [line.split(' ') for line in lines]
        calculated_systems = [line[0].strip() for line in lines]
else:
    calculated_systems = []

data_path = Path('data/bcurves')
for file in data_path.rglob('*'):
    if file.is_file() and file.suffix == '.xyz':
        if file.name not in energies or file.name in calculated_systems:
            continue

        charges = get_charges(file.name)
        
        filename = str(file)
        try:
            mono1_in_dimer_geom, mono2_in_dimer_geom, mono1_geom, mono2_geom = geom_from_xyz_dimer_ghosts(filename, charges)
        except TypeError:
            print(f'Failed to read dimer xyz for {file.name}')
            continue
       
        try:
            print(f'Starting calculation for {file.name}', flush=True)
            psi4.core.clean()
            cp_corr = cp_correction(mono1_in_dimer_geom, mono2_in_dimer_geom, mono1_geom, mono2_geom)
            final_energy = energies[file.name] - cp_corr 
            print(f'CP correction: {cp_corr}')
            print(f'Final energy: {final_energy}', flush=True)
 
            with open(cp_energies_path, 'a+') as fd:
                fd.write(f'{file.name} {final_energy}\n')

        except Exception as e:
            print(e)
            print(f'Failed to compute CP correction for {file.name}', flush=True)
            continue

