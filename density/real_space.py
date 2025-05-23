'''
Code modified from: https://github.com/semodi/mlcf/blob/master/mlc_func/elf/real_space.py
'''
import numpy as np
from scipy.special import sph_harm
import scipy.linalg
from sympy.physics.wigner import clebsch_gordan as CG
from sympy import N
from functools import reduce
import time
from ase import Atoms
from .density import Density
from ase.units import Bohr
from .geom import get_nncs_angles, get_elfcs_angles, get_casimir
from .geom import make_real, rotate_tensor, fold_back_coords, power_spectrum, transform
from .elf import ElF
from .serial_view import serial_view

def mesh_around(pos, radius, density, unit = 'A'):
    '''
    Similar to box_around but only returns mesh
    '''

    if pos.shape != (1,3) and (pos.ndim != 1 or len(pos) !=3):
        raise Exception('please provide only one point for pos')

    pos = pos.flatten()

    U = np.array(density.unitcell)  # Matrix to go from real space to mesh coordinates
    for i in range(3):
        U[i, :] = U[i, :]/density.grid[i]
    a = np.linalg.norm(density.unitcell, axis=1)/density.grid[:3]
    U = U.T

    # Create box with max. distance = radius
    rmax = np.ceil(radius / a).astype(int).tolist()
    Xm, Ym, Zm = density.mesh_3d(scaled = False, pbc= False, rmax = rmax, indexing = 'ij')
    X, Y, Z = density.mesh_3d(scaled = True, pbc= False, rmax = rmax, indexing = 'ij')

    # Find mesh pos.
    cm = np.round(np.linalg.inv(U).dot(pos)).astype(int)
    dr = pos  - U.dot(cm)
    X -= dr[0]
    Y -= dr[1]
    Z -= dr[2]

    Xm = (Xm + cm[0])%density.grid[0]
    Ym = (Ym + cm[1])%density.grid[1]
    Zm = (Zm + cm[2])%density.grid[2]

    return Xm, Ym, Zm

def mesh_around_nonperiodic(pos, radius, density, unit='A'):
    if pos.shape != (1,3) and (pos.ndim != 1 or len(pos) !=3):
        raise Exception('please provide only one point for pos')

    pos = pos.flatten()

    U = np.array(density.unitcell)  # Matrix to go from real space to mesh coordinates
    for i in range(3):
        U[i, :] = U[i, :]/density.grid[i]
    a = np.linalg.norm(density.unitcell, axis=1)/density.grid[:3]
    U = U.T

    # Create box with max. distance = radius from position
    rmax = np.ceil(radius/a).astype(int).tolist()
    Xm, Ym, Zm = density.mesh_3d_nonperiodic(scaled = False, rmax = rmax, indexing = 'ij')
    X, Y, Z = density.mesh_3d_nonperiodic(scaled = True, rmax = rmax, indexing = 'ij')

    # Find mesh position
    cm = np.round(np.linalg.inv(U).dot(pos).astype(int))


def box_around(pos, radius, density):
    '''
    Return dictionary containing box around an atom at position pos with
    given radius. Dictionary contains box in mesh, euclidean and spherical
    coordinates

    Parameters
    ---

        pos: np.ndarray (3,) or (1,3),
            coordinates for box center
        radius: float
            box radius
        density: Density
            only needed for Density.unitcell and Density.grid

    Returns
    ---

        dict
            {'mesh','real','radial'}, box in mesh,
            euclidean and spherical coordinates
    '''

    if pos.shape != (1,3) and (pos.ndim != 1 or len(pos) !=3):
        raise Exception('please provide only one point for pos')

    pos = pos.flatten()

    U = np.array(density.unitcell) # Matrix to go from real space to mesh coordinates
    for i in range(3):
        U[i,:] = U[i,:] / density.grid[i]
    a = np.linalg.norm(density.unitcell, axis = 1)/density.grid[:3]
    U = U.T

    #Create box with max. distance = radius
    rmax = np.ceil(radius / a).astype(int).tolist()
    Xm, Ym, Zm = density.mesh_3d(scaled = False, pbc= False, rmax = rmax, indexing = 'ij')
    X, Y, Z = density.mesh_3d(scaled = True, pbc= False, rmax = rmax, indexing = 'ij')

    #Find mesh pos.
    cm = np.round(np.linalg.inv(U).dot(pos)).astype(int)
    dr = pos  - U.dot(cm)
    X -= dr[0]
    Y -= dr[1]
    Z -= dr[2]

    Xm = (Xm + cm[0])%density.grid[0]
    Ym = (Ym + cm[1])%density.grid[1]
    Zm = (Zm + cm[2])%density.grid[2]

    R = np.sqrt(X**2 + Y**2 + Z**2)

    Phi = np.arctan2(Y,X)
    Theta = np.arccos(Z/R, where = (R != 0))
    Theta[R < 1e-15] = 0

    return {'mesh':[Xm, Ym, Zm],'real': [X,Y,Z],'radial':[R, Theta, Phi]}

def g(r, r_i, r_c, a, gamma):
    """
    Non-orthogonalized radial functions

    Parameters
    -------

        r: float
            radius
        r_i: float
            inner radial cutoff
        r_o: float
            outer radial cutoff
        a: int
            exponent (equiv. to radial index n)
        gamma: float
            damping parameter

    Returns
    ------

        float
            value of radial function at radius r
    """

    def g_(r, r_i, r_c, a):
        return (r-r_i)**(2)*(r_c-r)**(a+2)*np.exp(-gamma*(r/r_c)**(1/4))
#          return (r-r_i)**(5)*(r_c-r)**(a+2)
    r_grid = np.arange(r_i, r_c, (r_c-r_i)/1e3)
    N = np.sqrt(np.sum(g_(r_grid,r_i,r_c, a)*g_(r_grid,r_i,r_c,a))*(r_c-r_i)/1e3)
    return g_(r,r_i,r_c,a)/N

def S(r_i, r_o, nmax, gamma):
    '''
    Overlap matrix between radial basis functions

    Parameters
    -------

        r_i: float
            inner radial cutoff
        r_o: float
            outer radial cutoff
        nmax: int
            max. number of radial functions
        gamma: float
            damping parameter

    Returns
    -------

        np.ndarray (nmax, nmax)
            Overlap matrix
    '''

    S_matrix = np.zeros([nmax,nmax])
    r_grid = np.arange(r_i, r_o, (r_o-r_i)/1e3)
    for i in range(nmax):
        for j in range(i,nmax):
            S_matrix[i,j] = np.sum(g(r_grid,r_i,r_o,i+1, gamma)*g(r_grid,r_i,r_o,j+1, gamma))*(r_o-r_i)/1e3
    for i in range(nmax):
        for j in range(i+1, nmax):
            S_matrix[j,i] = S_matrix[i,j]
    return S_matrix


def radials(r, r_i, r_o, W, gamma):
    '''
    Get orthonormal radial basis functions

    Parameters
    -------

        r: float
            radius
        r_i: float
            inner radial cutoff
        r_o: float
            outer radial cutoff
        W: np.ndarray
            orthogonalization matrix
        gamma: float
            damping parameter

    Returns
    -------

        np.ndarray
            radial functions
    '''

    result = np.zeros([len(W)] + list(r.shape))
    for k in range(0,len(W)):
        rad = g(r,r_i,r_o,k+1, gamma)
        for j in range(0, len(W)):
            result[j] += W[j,k] * rad
    result[:,r > r_o] = 0
    result[:,r < r_i] = 0
#    result = result[::-1] # Invert so that n = 0 is closest to origin
    return result

def get_W(r_i, r_o, n, gamma):
    '''
    Get matrix to orthonormalize radial basis functions

    Parameters
    -------

        r_i: float
            inner radial cutoff
        r_o: float
            outer radial cutoff
        n: int
            max. number of radial functions
        gamma: float
            damping parameter

    Returns
    -------
        np.ndarray
            W, orthogonalization matrix
    '''
    return scipy.linalg.sqrtm(np.linalg.pinv(S(r_i,r_o, n, gamma)))

def decompose(rho, box, n_rad, n_l, r_i, r_o, gamma, V_cell = 1):
    '''
    Project the real space density rho onto a set of basis functions

    Parameters
    ----------
        rho: np.ndarray
            electron charge density on grid
        box: dict
             contains the mesh in spherical and euclidean coordinates,
             can be obtained with get_box_around()
        n_rad: int
             number of radial functions
        n_l: int
             number of spherical harmonics
        r_i: float
             inner radial cutoff in Angstrom
        r_o: float
             outer radial cutoff in Angstrom
        gamma: float
             exponential damping
        V_cell: float
             volume of one grid cell

    Returns
    --------
        dict
            dictionary containing the complex ELF
    '''

    R, Theta, Phi = box['radial']
    Xm, Ym, Zm = box['mesh']

    # Automatically detect whether entire charge density or only surrounding
    # box was provided
    if rho.shape == Xm.shape:
        small_rho = True
    else:
        small_rho = False

    #Build angular part of basis functions
    angs = []
    for l in range(n_l):
        angs.append([])
        for m in range(-l,l+1):
            # angs[l].append(sph_harm(m, l, Phi, Theta).conj()) TODO: In theory should be conj!?
            angs[l].append(sph_harm(m, l, Phi, Theta))
    

    #Build radial part of b.f.
    W = get_W(r_i, r_o, n_rad, gamma) # Matrix to orthogonalize radial basis
    rads = radials(R, r_i, r_o, W, gamma)

    coeff = {}
    if small_rho:
        for n in range(n_rad):
            for l in range(n_l):
                for m in range(2*l+1):
                    coeff['{},{},{}'.format(n,l,m-l)] = np.sum(angs[l][m]*rads[n]*rho)*V_cell
    else:
        for n in range(n_rad):
            for l in range(n_l):
                for m in range(2*l+1):
                    coeff['{},{},{}'.format(n,l,m-l)] = np.sum(angs[l][m]*rads[n]*rho[Xm, Ym, Zm])*V_cell
    return coeff

def atomic_elf(pos, density, basis, chem_symbol):
    """Given an input density and an atomic position decompose the
    surrounding charge density into an ELF

    Parameters
    ----------
        pos: (,3) np.ndarray
            atomic position
        density: Density
            stores charge density rho, unitcell, and grid (see density.py)
        basis: dict
            specifies the basis set used for the ELF decomposition for each chem. element
        chem_symbol: str
            chemical element symbol

    Returns
    --------
        dict
            dictionary containing the real ELF
    """

    chem_symbol = chem_symbol.lower()

    if pos.shape == (3,):
        pos = pos.reshape(1,3)
    if pos.shape != (1,3):
        raise Exception('pos has wrong shape')

    U = np.array(density.unitcell) # Matrix to go from real space to mesh coordinates
    for i in range(3):
        U[i,:] = U[i,:] / density.grid[i]
    V_cell = np.linalg.det(U)

    # The following two lines are needed to
    #obtain the dataset from the old implementation
    V_cell /= (37.7945/216)**3*Bohr**3
    V_cell *= np.sqrt(Bohr)

    box = box_around(pos, basis['r_o_' + chem_symbol], density)
    coeff = decompose(density.rho, box,
                           basis['n_rad_' + chem_symbol],
                           basis['n_l_' + chem_symbol],
                           basis['r_i_' + chem_symbol],
                           basis['r_o_' + chem_symbol],
                           basis['gamma_' + chem_symbol],
                           V_cell = V_cell) # change this to xs * xy *xz = 0.93 later
    return coeff

def get_elf_thread(pos, density, basis, chem_symbol,
    i, all_positions, mode):
    """ Method that should be used in a parallel executions.
    One thread/process computes and orients the ElF for a single atom
    inside a system
    """

    values = list(map(atomic_elf, pos, density, basis, chem_symbol))
    elfs = [ElF(v,[0,0,0],b,
        c,d.unitcell) for v,b,c,d in zip(values, basis, chem_symbol, density)] # no orientation

    if not mode == 'none':
        elf_oriented = list(map(orient_elf,i,elfs,[all_positions]*len(elfs),
            [mode]*len(elfs)))
        return elf_oriented
    else:
        return elfs


def get_elfs(atoms, density, basis, view = serial_view(), orient_mode = 'none'):
    '''
    Given an input density and an ASE Atoms object decompose the
    complete charge density into atomic ELFs

    Parameters
    ----------
        atoms: ase.Atoms
        density: Density
            stores charge density rho, unitcell, and grid (see density.py)
        basis: dict
            specifies the basis set used for the ELF decomposition for each chem. element

        view: ipyparallel.balanced_view
            for parallel execution through sync map

        orient_mode: str
            {'none': do not orient and return complex tensor,
           'elf'/'nn': orient using the elf or nn algorithm and return
           real tensor}
    Returns
    --------
        list
            list containing the complex/real atomic ELFs '''

    def distribute_workload(array, n_workers):
        job_list = []
        for w in range(n_workers):
            job_list.append(array[w::n_workers])
        return job_list

    basis_species = np.unique([b[-1] for b in basis\
        if b[-2] == '_'])

    density_list = []
    pos_list = []
    sym_list = []
    basis_list = []
    indices_list = []
    n_workers = len(view)
    spec_filt = [sp.lower() in basis_species for sp in atoms.get_chemical_symbols()]
    for pos, sym, idx in zip(distribute_workload(atoms.get_positions()[spec_filt], n_workers),
                        distribute_workload(np.array(atoms.get_chemical_symbols())[spec_filt], n_workers),
                        distribute_workload(np.arange(len(atoms)).astype(int)[spec_filt], n_workers)):

        density_list.append([])
        pos_list.append([])
        basis_list.append([])
        sym_list.append([])
        indices_list.append([])
        for p, s, i in zip(pos, sym, idx):
            if s.lower() not in basis_species: continue   # Skip atoms for which no basis provided
            mesh = mesh_around(p, basis['r_o_' + s.lower()], density)
            density_list[-1].append(Density(density.rho[mesh],
                                                density.unitcell,
                                                density.grid, density.origin))
            pos_list[-1].append(p)
            sym_list[-1].append(s)
            basis_list[-1].append(basis)
            indices_list[-1].append(i)

    n_jobs = len(basis_list)
    all_pos = atoms.get_positions()

    elfs = view.map_sync(get_elf_thread, pos_list, density_list,
      basis_list, sym_list, indices_list,[all_pos]*n_jobs,
      [orient_mode]*n_jobs)

    elfs_flat = []

    max_len = len(elfs[0])
    for i in range(max_len):
        for e in elfs:
            try:
                elfs_flat.append(e.pop(0))
            except IndexError:
                break
        else:
            continue
        break

    return elfs_flat

def get_elfs_oriented(atoms, density, basis, mode, view = serial_view()):
    """Outdated, use get_elfs() with "mode='elf'/'nn'" instead.
    Like get_elfs, but returns real, oriented elfs
    mode = {'elf': Use the ElF algorithm to orient fingerprint,
            'nn': Use nearest neighbor algorithm,
            'casimir'}
    """
    return get_elfs(atoms, density, basis, view, orient_mode = mode)

def orient_elf(i, elf, all_pos, mode):
    '''
    Takes an ELF and orients it according to the rule specified in mode.

    Parameters
        i: int
            index of the atom in all_pos
        elf: ELF
            ELF to orient
        all_pos: np.ndarray
            positions of all atoms in the system
        mode: str
            {'elf' : use the ELF algorithm to orient the fingerprint
            'nn': use the nearest neighbour algorithm
            'casimir': take Casimir norm of complex tensor
            'neutral': keep alignment unchanged}
    
    Returns
        ELF
            oriented version of elf
    '''
    if mode == 'elf':
        angles_getter = get_elfcs_angles
    elif mode == 'nn':
        angles_getter = get_nncs_angles
    elif mode == 'neutral':
        pass
    elif mode == 'casimir':
        pass
    elif mode == 'power_spectrum':
        pass
    else:
        raise Exception('Unknown!! orientation mode {}'.format(mode))

    if (mode.lower() == 'neutral') or (mode == 'casimir') or (mode == 'power_spectrum'):
        angles = np.array([0,0,0])
    else:
        angles = angles_getter(i, fold_back_coords(i, all_pos, elf.unitcell), elf.value)

    if mode == 'casimir':
        oriented = get_casimir(elf.value)
        oriented = np.asarray(list(oriented.values()))
        elf_oriented = ElF(oriented, angles, elf.basis, elf.species, elf.unitcell)
    elif mode == 'neutral':
        oriented = make_real(rotate_tensor(elf.value, np.array(angles), True))
        elf_oriented = ElF(oriented, angles, elf.basis, elf.species, elf.unitcell)
    else:
        elf_transformed = transform(elf.value)
        elf_transformed = np.stack([val for val in elf_transformed.values()]).reshape(-1)
        n_l = elf.basis[f'n_l_{elf.species.lower()}']
        n = elf.basis[f'n_rad_{elf.species.lower()}']
        ps = power_spectrum(elf_transformed.reshape(1,-1), n_l-1, n, cgs=None)
        oriented = ps.reshape(-1)
        elf_oriented = ElF(oriented, angles, elf.basis, elf.species, elf.unitcell)
    return elf_oriented

def orient_elfs(elfs, atoms, mode):
    """Convenience function that applies orient_elf to a list of elfs.
       (Exists for compatibility reasons)
    """

    oriented_elfs = []
    for i, elf in enumerate(elfs):
        oriented_elfs.append(orient_elf(i ,elf, atoms.get_positions(),mode))

    return oriented_elfs
