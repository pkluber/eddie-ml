from pathlib import Path
import tarfile, os

from density.cube_utils import xyz_to_Mol, dimerxyz_to_Mol

from pyscf import scf, mp
from pyscf.gto import Mole
from pyscf.lib import config

NEUTRAL_TAR = 'data/neutral-dimers.tar.gz' 

def list_contents(tarfile_path: str) -> list[str]:
    outputs = []
    with tarfile.open(tarfile_path, 'r:gz') as tar:
        for member in tar.getmembers():
            filename = os.path.basename(member.name)
            outputs.append(filename)

    return outputs


NEUTRAL_SYSTEMS = list_contents(NEUTRAL_TAR)

if config.use_gpu:
    print('GPU is enabled!')
else:
    print('GPU is not enabled :C')

def gpuify(x):
    if config.use_gpu:
        return x.to_gpu()
    return x

def mp2(mol: Mole) -> mp.MP2:
    mf = gpuify(scf.RHF(mol))
    mf.kernel()
    mp2 = gpuify(mp.MP2(mf))
    mp2.kernel()
    return mp2
    

def mp2_energy(mol: Mole):
    return mp2(mol).e_tot

def srs_mp2_int_energy(dimer: Mole, mono1: Mole, mono2: Mole):
    # Ensure basis is cc-pVTZ for SRS-MP2
    dimer.basis = mono1.basis = mono2.basis = 'cc-pvtz' 
    
    d_mp2 = mp2(dimer), m1_mp2 = mp2(mono1), m2_mp2 = mp2(mono2)
    
    e_os_int = d_mp2.e_corr_os - m1_mp2.e_corr_os - m2_mp2.e_corr_os
    e_ss_int = d_mp2.e_corr_ss - m1_mp2.e_corr_ss - m2_mp2.e_corr_ss
    eps_ds = e_os_int / e_ss_int

    if eps_ds >= 1:
        c_os = 1.640, c_ss = 0
    else:
        c_os = 0.660, c_ss = 1.140

    uncorr_int = d_mp2._scf.e_tot - m1_mp2._scf.e_tot - m2_mp2._scf.e_tot
    
    return uncorr_int + c_os * e_os_int + c_ss * e_ss_int 


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
            spins = 0, 0, 0
        elif file.name.startswith('C'): # IL174 
            charges = 0, 1, -1
            spins = 0, 0, 0
        else: # other charged part of SSI
            continue

        tot_charge, mon1_charge, mon2_charge = charges 
        tot_spin, mon1_spin, mon2_spin = spins

        filename = str(file)
        try:
            dimer = dimerxyz_to_Mol(filename, charge=tot_charge, spin=tot_spin)
            mono1 = xyz_to_Mol(filename, n=0, charge=mon1_charge, spin=mon1_spin)
            mono2 = xyz_to_Mol(filename, n=1, charge=mon2_charge, spin=mon2_spin)
        except RuntimeError as e:
            print(f'Failed to initialize Mole objects for {file.name}: probably spin is wrong')
            print(e)
            continue
        
        try:
            print(f'Running calculations for {file.name}', flush=True)
            interaction_energy = srs_mp2_int_energy(dimer, mono1, mono2)
            with open('energies.dat', 'a+') as fd:
                fd.write(f'{file.name} {interaction_energy}\n')
            print(f'Finished calculations for {file.name}!', flush=True)

        except:
            print(f'Failed calculations for {file.name}', flush=True)
            continue
