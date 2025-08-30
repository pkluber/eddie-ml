import h5py

# Open the HDF5 file
with h5py.File('output.hdf5', 'r') as f:
    # List all groups and datasets in the file
    print("Keys in file: %s" % list(f.keys()))
    
    X = f['system'][:]
    species = f['species'][:]
    positions = f['position'][:]

#with h5py.File('output_pca.hdf5', 'r') as fd:
#    Vs = fd['value'][:]
#    print(list(fd.keys()))
#    print(Vs.shape)

# Parse energies.dat
energies = {}
with open('energies.dat') as fd:
    lines = fd.readlines()
    for line in lines:
        xyz, ie = line.split(' ')
        system_name = xyz[:-4]
        ie = float(ie)
    
        energies[system_name] = ie

print(list(energies.items())[0])

test_str = X[0].decode('ascii')
test_species = species[0].decode('ascii')
print(test_str, test_species, energies[test_str])
