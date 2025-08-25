from pathlib import Path
import tarfile, os

import psi4

psi4.core.set_num_threads(48)
psi4.set_memory('80 GB')

NEUTRAL_TAR = 'data/neutral-dimers.tar.gz' 

def list_contents(tarfile_path: str) -> list[str]:
    outputs = []
    with tarfile.open(tarfile_path, 'r:gz') as tar:
        for member in tar.getmembers():
            filename = os.path.basename(member.name)
            outputs.append(filename)

    return outputs


NEUTRAL_SYSTEMS = list_contents(NEUTRAL_TAR)

def mp2(geom: str) -> tuple[float, float, psi4.core.Wavefunction]:
    mol = psi4.geometry(geom)
    psi4.set_options({'basis': 'aug-cc-pvqz'})
    hf_e, hf_wfn = psi4.energy('scf', return_wfn=True)
    psi4.set_options({'basis': 'cc-pvtz'})
    mp2_e, mp2_wfn = psi4.energy('mp2', mp2_type='df', ref_wfn=hf_wfn, return_wfn=True)
    return hf_e, mp2_e, mp2_wfn
    
MP2_OS_ENG = 'MP2 OPPOSITE-SPIN CORRELATION ENERGY'
MP2_SS_ENG = 'MP2 SAME-SPIN CORRELATION ENERGY'

def srs_mp2_int_energy(dimer_geom: str, mono1_geom: str, mono2_geom: str): 
    d_hf_e, d_mp2_e, d_wfn = mp2(dimer_geom) 
    m1_hf_e, m1_mp2_e, m1_wfn = mp2(mono1_geom)
    m2_hf_e, m2_mp2_e, m2_wfn = mp2(mono2_geom)
        
    d_corr_os = d_wfn.variable(MP2_OS_ENG)
    d_corr_ss = d_wfn.variable(MP2_SS_ENG)
    m1_corr_os = m1_wfn.variable(MP2_OS_ENG)
    m1_corr_ss = m1_wfn.variable(MP2_SS_ENG)
    m2_corr_os = m2_wfn.variable(MP2_OS_ENG)
    m2_corr_ss = m2_wfn.variable(MP2_SS_ENG)

    e_os_int = d_corr_os - m1_corr_os - m2_corr_os
    e_ss_int = d_corr_ss - m1_corr_ss - m2_corr_ss
    eps_ds = e_os_int / e_ss_int

    if eps_ds >= 1:
        c_os = 1.640
        c_ss = 0
    else:
        c_os = 0.660
        c_ss = 1.140

    uncorr_int = d_hf_e - m1_hf_e - m2_hf_e
    
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
        

data_dir = Path('data/bcurves')
for file in data_dir.rglob('*'):
    if file.is_file() and file.suffix == '.xyz':
        # Skip over files that alread yappear in energies.dat
        if file.name in calculated_systems:
            continue

        # Determine charges
        if file.name in NEUTRAL_SYSTEMS: # S66, part of SSI
            charges = 0, 0, 0
        elif file.name.startswith('C'): # IL174, extraILs 
            charges = 0, 1, -1
        else: # other charged part of SSI
            continue

        tot_charge, mon1_charge, mon2_charge = charges 

        filename = str(file)
        try:
            psi4.core.clean()
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

        except Exception as e:
            print(e)
            print(f'Failed calculations for {file.name}', flush=True)
            continue
