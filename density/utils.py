'''
Code modified from: https://github.com/semodi/mlcf/blob/master/mlc_func/elf/utils.py
'''
import h5py
from h5py import File
import json
import numpy as np
from ase import Atoms
from ase.io import read, write
import os
from .elf import ElF
import ipyparallel as ipp
import re
import pandas as pd
from .read_cubes import load_cube, get_atoms
from .real_space import get_elfs_oriented, orient_elfs
from .geom import make_complex, rotate_tensor
from .serial_view import serial_view

def get_view(profile = 'default', n = -1):
    """
    Load ipyparallel balanced_view

    Parameters
    ---
        profile: str
            ipyparallel profile, default='default'
        n: int
            use n workers, default=-1 (use all workers)

    Returns
    ----
        ipyparallel.balanced_view

    """
    client = ipp.Client(profile = profile)
    # view = client.load_balanced_view()
    if n == -1:
        view = client[:]
    else:
        view = client[:n]
    print('Clients operating : {}'.format(len(client.ids)))
    #n_clients = len(client.ids)
    #print(n_clients)
    return view

def __get_elfs(path, atoms, basis, method):
    try:
        density = load_cube(path)
    except UnicodeDecodeError:
        density = get_density_bin(path)
    return get_elfs_oriented(atoms, density, basis, method)


def __get_all(paths, method, basis, eng_ext, dens_ext, n_atoms):
    atoms = list(map(get_atoms,[p + '.' + dens_ext for p in paths])) #,[n_atoms]*len(paths)))
    elfs = list(map(__get_elfs, [p + '.' + dens_ext for p in paths],
     atoms, [basis]*len(paths), [method]*len(paths)))

    return atoms, elfs

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def preprocess_all(root, basis, name=None, dens_ext = 'cube', eng_ext='xyz',
    method = 'elf', view = serial_view(), n_atoms = -1, walk=True):
    """ Given a root directory, walk directory tree and find all files ending in dens_ext and transform
    electron density into descriptors.

    Parameters
    ----------
        root: str
            root directory containing all data
        basis: dict
            basis to use to create descriptors
        dens_ext: str
            extension for electron density files (atomic structural info is also read from here)
        eng_ext: str
            extension of file containing energy values (modify for later)
        method: str
            {'elf','nn','water','neutral','casimir'}, pick alignment method
        view: ipyparallel.balanced_view
            for parallel processing
        n_atoms: int
            only process the first n_atoms atoms in each system,
            default: -1 (all atoms)
        walk: True
            if False, only uses the files in the current directory (doesn't search in subdirs).

    Returns
    --------
        list of ElF
            containing the created descriptors

    Other
    -------
        Saves energies, forces in a csv file, the structural information in an ase .traj file,
        the descriptors in a .hdf5 file
    """
    
    if isinstance(root, list):
        for i in range(len(root)):
            if root[i][-1] == '/':
                root[i] = root[i][:-1]
            if root[i][0] != '/' and root[i][0] != '~':
                root[i] = os.getcwd() + '/' + root[i]

    else: # root is a string (one directory only)
        if root[-1] == '/':
            root = root[:-1]
        if root[0] != '/' and root[0] != '~':
            root = os.getcwd() + '/' + root

    if walk and isinstance(root, list):
        raise Exception('Can only have walk=True if root is single directory.')
    
    paths = []

    if walk:
        for branch in os.walk(root):
            files = np.unique([t.split('.')[0] for t in branch[2] if (t.endswith(dens_ext) or t.endswith(eng_ext))])
            paths += [branch[0] + '/' + f for f in files]
    else:
        if isinstance(root, list):
            for i in range(len(root)):
                files = np.unique([t.split('.')[0] for t in os.listdir(root[i]) if (t.endswith(dens_ext) or t.endswith(eng_ext))])
                paths += [root[i] + '/' + f for f in files]
        else:
            files = np.unique([t.split('.')[0] for t in os.listdir(root) if (t.endswith(dens_ext) or t.endswith(eng_ext))])
            paths += [root + '/' + f for f in files]

    # Sort path for custom directory structure node_*/*.ext

    paths = sorted(paths, key=natural_keys)

    print(paths)
    print('{} systems found. Processing ...'.format(len(paths)))

    n_workers = len(view)
    print("Clients working = ", n_workers)
    full_workload = len(paths)

    min_workload = np.floor(full_workload/n_workers).astype(int)
    max_workload = min_workload + 1
    n_max_workers = full_workload - min_workload*n_workers

    paths_dist = [paths[i*(max_workload):(i+1)*(max_workload)] for i in range(n_max_workers)]

    offset = n_max_workers*max_workload

    paths_dist += [paths[offset + i*min_workload:offset + (i+1) * min_workload] for i in range(n_workers - n_max_workers)]

    all_results = list(view.map(__get_all,
     paths_dist, [method]*len(paths_dist), [basis]*len(paths_dist),
      [eng_ext]*len(paths_dist), [dens_ext]*len(paths_dist), [n_atoms]*len(paths_dist)))
    atoms, elfs = list(map(list, zip(*all_results)))

    elfs = [e for sublist in elfs for e in sublist]
    atoms = [a for sublist in atoms for a in sublist]
    if name == None:
        name = root.split('/')[-1]
    else:
        name = str(name)
    elfs_to_hdf5(elfs, name + '.hdf5', paths)
    write(name +'.traj', atoms)
    return elfs

def elfs_to_hdf5(elfs, path, paths):
    """
    Given a list of electronic desriptors (ElFs) save them in an .hdf5 file

    Parameters
    ----------

        elfs: list of ElFs
            to save
        path: str
            file destination
        paths: list[str]
            list of paths

    Returns
    --------
        None
    """

    file = h5py.File(path, 'w')
    max_len = 0
    full_basis = {}
    system_label = ''
    for atom in elfs[0]:
        max_len = max([max_len, len(atom.value)])
        system_label += atom.species
        for b in atom.basis:
            full_basis[b] = atom.basis[b]

    file.attrs['basis'] = json.dumps(full_basis)
    file.attrs['system'] = system_label
    values = []
    lengths = []
    species = []
    angles = []
    systems = []

    for s, system in enumerate(elfs):
        for a, atom in enumerate(system):
            v = atom.value
            lengths.append(len(v))
            if len(v) != max_len:
                v_long = np.zeros(max_len)
                v_long[:len(v)] = v
                v = v_long
            values.append(v)
            angles.append(atom.angles)
            species.append(atom.species.encode('ascii','ignore'))
            systems.append(os.path.basename(paths[s]))

    file['value'] = np.array(values)
    file['length'] = np.array(lengths)
    file['species'] = species
    file['angles'] = np.array(angles)
    file['system'] = systems
    file.flush()

def hdf5_to_elfs(path, species_filter = '', grouped = False,
        values_only = False, angles_only = False):
    """ Loads .hdf5 file that stores electronic descriptors (ElFs) and returns them
    Parameters
    -----------
        path: str
        file origin
        species_filter: str
            only load elements specified in this string
        grouped: boolean
            return a dictionary that groups values by element
        values_only: boolean
            only return the descriptor values (tensor)
        angles_only: boolean
            only return the descriptor angles
    Returns:
    --------
        list of ElFs / dict (if grouped = True) / np.ndarray (if either values_only or angles_only = True)
            ElFs from hdf5 file
    """

    file = h5py.File(path, 'r')
    basis = json.loads(file.attrs['basis'])
    print(basis)
    if values_only and angles_only:
        raise Exception('Cannot return values and angles only at the same time')
    if values_only or angles_only:
        grouped = True

    if grouped:
        elfs_grouped = {}
    else:
        elfs = []

    if grouped:
        current_system_grouped = {}
    else:
        current_system = -1

    for value, length, species, angles, system in zip(file['value'][:],
                                                  file['length'][:],
                                                  file['species'][:],
                                                  file['angles'][:],
                                                  file['system'][:]):
        species = species.astype(str).lower()
        if len(species_filter) > 0 and\
         (not (species in species_filter.lower())):
            continue

        if grouped:
            if not species in elfs_grouped:
                elfs_grouped[species] = []
                current_system_grouped[species] = -1
            elfs = elfs_grouped[species]
            current_system = current_system_grouped[species]

        if system != current_system:
            elfs.append([])
            if grouped:
                current_system_grouped[species] = system
            else:
                current_system = system

        bas =  dict(filter(lambda x: species in x[0].lower(), basis.items()))

        if values_only:
            elfs[system].append(value[:length])
        elif angles_only:
            elfs[system].append(angles)
        else:
            elfs[system].append(ElF(value[:length], angles, bas, species, np.zeros(3)))

    if grouped:
        elfs = elfs_grouped
    return elfs


def hdf5_to_elfs_fast(path, species_filter = ''):
    """ Loads .hdf5 file that stores electronic descriptors (ElFs) and returns themself.
    Faster implementation of hdf5_to_elfs. Only works for homogeneous datasets.

    Parameters
    -----------

        path: str
            file origin
        species_filter: str
            only load elements specified in this string

    Returns
    --------

        (dict, dict)
            containing values and angles for each element respectively

    """

    file = h5py.File(path, 'r')
    basis = json.loads(file.attrs['basis'])
    print(basis)

    values_dict = {}
    angles_dict = {}
    values = file['value'][:]
    angles = file['angles'][:]
    all_species = file['species'][:]
    all_lengths = file['length'][:]
    systems = file['system'][:]
    unique_systems, count_system = np.unique(systems,return_counts=True)
    if not len(np.unique(count_system)) == 1:
        raise Exception('Dataset not homogeneous, use hdf5_to_elfs() instead')
    else:
        n_systems = len(unique_systems)
    if len(species_filter) == 0:
        species_filter = [s.astype(str).lower() for s in np.unique(all_species)]

    for species in species_filter:
        filt = ((all_species.astype(str) == species.upper()) | (all_species.astype(str) == species.lower()))
        length = all_lengths[np.where(filt)[0][0]]
        values_dict[species] = values[filt,:length].reshape(n_systems,-1,length)
        angles_dict[species] = angles[filt,:].reshape(n_systems,-1,3)

    return values_dict, angles_dict

def elfs_to_descriptor(filename, pad=False, save_npy=True):
    '''
    Reads in the ELFs contained in the filename and returns an array X of the
    corresponding descriptors.
    When pad = True, pads the ELFs to have the same number of atoms as the largest
    system in the dataset (i.e., fills empty atoms spots with rows of zeros.)
    '''
    X = []
    file = h5py.File(filename, 'r')
    elfs = hdf5_to_elfs(filename, grouped=False) # returns elfs for each atomic system

    max_len = np.max(np.unique(file['length'][:])) # find max ELF length for padding
    unique_systems, count_system = np.unique(file['system'][:], return_counts=True)
    max_atoms = np.max(count_system)

    for i, system in enumerate(elfs):
        if pad:
            x = np.zeros((max_atoms, max_len))
        else:
            x = np.zeros((len(system), max_len)) # array (number of atoms, max_len)

        for a, atom in enumerate(system):
            if len(atom.value) != max_len:
                x[a] = np.pad(atom.value, (0, max_len - len(atom.value))) # pad end with zeros
            else:
                x[a] = atom.value

        X.append(x)

    X = np.asarray(X)

    if save_npy:
        np.save('descriptor.npy', X)
    return X

def change_alignment(path, traj_path, new_method, save_as = None):
    """ Transform ElFs from one local coordinate system to another one specified in
    new_method

    Parameters
    ----------
        path: str
            file origin
        traj_path: str
            path to .traj or .xyz containing the atomic positions
        new_method: str
            {'elf','nn','water','neutral'}, method defining the local coordinate system
        save_as: str
            destination file to save ElFs, default=None (don't save)

    Returns
    ---------
        list of ElF
            if save_as == None returns list of electronic descriptors (ElFs)
            otherwise returns None

    """

    elfs = hdf5_to_elfs(path)
    atoms = read(traj_path, ':')

    with File(path) as file:
        basis = json.loads(file.attrs['basis'])

    # if new_method == basis['alignment']:
        # raise Exception('Already aligned with method: ' + new_method)
    # else:
    basis['alignment'] = new_method

    #Rotate to neutral
    for i, elf_system in enumerate(elfs):
        for j, elf in enumerate(elf_system):
            elfs[i][j].value = rotate_tensor(make_complex(elf.value, basis['n_rad_' + elf.species.lower()],
                                                       basis['n_l_' + elf.species.lower()]), elf.angles,
                                             inverse = False)
            elfs[i][j].angles = np.array([0,0,0])
            elfs[i][j].unitcell = atoms[i].get_cell()
            elfs[i][j].basis = basis

    oriented_elfs = []

    for elfs_system, atoms_system in zip(elfs, atoms):
        oriented_elfs.append(orient_elfs(elfs_system,atoms_system,new_method))

    if save_as == None:
        return oriented_elfs
    else:
        elfs_to_hdf5(oriented_elfs, save_as)
