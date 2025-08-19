from pathlib import Path
import tarfile, os

from density.cube_utils import xyz_to_Mol, dimerxyz_to_Mol

from ase.io import read
import psi4

psi4.core.set_num_threads(48)
psi4.core.set_memory('20 GB')

NEUTRAL_TAR = 'data/neutral-dimers.tar.gz' 

def list_contents(tarfile_path: str) -> list[str]:
    outputs = []
    with tarfile.open(tarfile_path, 'r:gz') as tar:
        for member in tar.getmembers():
            filename = os.path.basename(member.name)
            outputs.append(filename)

    return outputs


NEUTRAL_SYSTEMS = list_contents(NEUTRAL_TAR)

def mp2(geom: str) -> float:
    mol = psi4.geometry(geom)
    psi4.set_options({'basis': 'aug-cc-pvqz'})
    hf_e, hf_wfn = psi4.energy("scf", return_wfn=True)
    psi4.set_options({'basis': 'cc-pvtz'})
    mp2_e, mp2_wfn = psi4.energy("mp2", ref_wfn=hf_wfn, return_wfn=True)
    return mp2_e
    
def srs_mp2_int_energy(dimer_geom: str, mono1_geom: str, mono2_geom: str):
    # Ensure basis is cc-pVTZ for SRS-MP2
    dimer.basis = mono1.basis = mono2.basis = 'cc-pvtz' 
    
    d_mp2 = mp2(dimer) 
    m1_mp2 = mp2(mono1)
    m2_mp2 = mp2(mono2)
    
    e_os_int = d_mp2.e_corr_os - m1_mp2.e_corr_os - m2_mp2.e_corr_os
    e_ss_int = d_mp2.e_corr_ss - m1_mp2.e_corr_ss - m2_mp2.e_corr_ss
    eps_ds = e_os_int / e_ss_int

    if eps_ds >= 1:
        c_os = 1.640
        c_ss = 0
    else:
        c_os = 0.660
        c_ss = 1.140

    uncorr_int = d_mp2._scf.e_tot - m1_mp2._scf.e_tot - m2_mp2._scf.e_tot
    
    return uncorr_int + c_os * e_os_int + c_ss * e_ss_int 


ENERGY_FILE = Path('energies.dat')
calculated_systems = []
if ENERGY_FILE.is_file() and ENERGY_FILE.exists():
    with open(ENERGY_FILE, 'r') as fd:
        energies = fd.readlines()
        calculated_systems = [line.split(' ')[0] for line in energies]

def geom_from_xyz_dimer(filename: str, charges: tuple[int, int, int]) -> tuple[str, str, str] | None:
    with open(filename) as fd:
        lines = fd.readlines() # note preserves \n characters 
        try:
            num_atoms_m1 = int(lines[0])
            num_atoms_m2 = int(lines[num_atoms_m1+1])
            
            m1_start = 1
            geometry_m1 = "".join(lines[m1_start:m1_start+num_atoms_m1])
            m2_start = num_atoms_m1 + 2
            geometry_m2 = "".join(lines[m2_start:m2_start+num_atoms_m2])

            return f'{charges[0]} 0\n{geometry_m1+geometry_m2}'
                    f'{charges[1]} 0\n{geometry_m1}', \
                    f'{charges[2]} 0\n{geometry_m2}', \
        except ValueError:
            print(f'Error parsing xyz file {filename}')
            return None
        

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
            charges = 0, 1, -1
        else: # other charged part of SSI
            continue

        tot_charge, mon1_charge, mon2_charge = charges 

        filename = str(file)
        try:
            dimer_geom, mono1_geom, mono2_geom = geom_from_xyz_dimer(filename, charges)
        except TypeError:
            print(f'Failed to read dimer xyz for {file.name}')
            continue
        
        try:
            print(f'Running calculations for {file.name}', flush=True)
            interaction_energy = srs_mp2_int_energy(dimer_geom, mono1_geom, mono2_geom)
            with open('energies.dat', 'a+') as fd:
                fd.write(f'{file.name} {interaction_energy}\n')
            print(f'Finished calculations for {file.name}!', flush=True)

        except:
            print(f'Failed calculations for {file.name}', flush=True)
            continue
