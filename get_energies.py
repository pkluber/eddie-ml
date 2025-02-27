from pathlib import Path
import tarfile, os

from density.cube_utils import xyz_to_Mol, dimerxyz_to_Mol

from pyscf import scf, mp
from pyscf.gto import Mole

NEUTRAL_TAR = 'data/neutral-dimers.tar.gz' 

def list_contents(tarfile_path: str) -> list[str]:
    outputs = []
    with tarfile.open(tarfile_path, 'r:gz') as tar:
        for member in tar.getmembers():
            filename = os.path.basename(member.name)
            outputs.append(filename)

    return outputs

NEUTRAL_SYSTEMS = list_contents(NEUTRAL_TAR)

def mp2_energy(mol: Mole):
    mf = scf.RHF(mol)
    mf.kernel()
    mp2 = mp.MP2(mf)
    mp2.kernel()
    return mp2.e_tot

ENERGY_FILE = Path('energies.dat')
calculated_systems = []
if ENERGY_FILE.is_file() and ENERGY_FILE.exists():
    with open(ENERGY_FILE, 'r') as fd:
        energies = fd.readlines()
        calculated_systems = [line.split(' ')[0] for line in energies]


data_dir = Path('data/bcurves')
for file in data_dir.rglob('*'):
    if file.is_file() and file.suffix == '.xyz':
        # Skip over files that alread yappear in energies.dat
        if file.name in calculated_systems:
            continue

        # Determine charges
        if file.name in NEUTRAL_SYSTEMS: # S66, part of SSI
            charges = 0, 0, 0
        elif file.name.startswith('C'): # IL174 
            charges = 1, -1, 0
        else: # other charged part of SSI
            continue

        tot_charge, mon1_charge, mon2_charge = charges 

        filename = str(file)
        try:
            dimer = dimerxyz_to_Mol(filename, charge=tot_charge)
            mono1 = xyz_to_Mol(filename, n=0, charge=mon1_charge)
            mono2 = xyz_to_Mol(filename, n=1, charge=mon2_charge)
        except RuntimeError:
            print(f'Failed to initialize Mole objects for {file.name}: probably spin is wrong')
            continue
        
        try:
            print(f'Running calculations for {file.name}', flush=True)
            interaction_energy = mp2_energy(dimer) - mp2_energy(mono1) - mp2_energy(mono2)
            with open('energies.dat', 'a+') as fd:
                fd.write(f'{file.name} {interaction_energy}\n')
            print(f'Finished calculations for {file.name}!', flush=True)

        except:
            print(f'Failed calculations for {file.name}', flush=True)
            continue
