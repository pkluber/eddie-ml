import h5py

# Open the HDF5 file
with h5py.File('output.hdf5', 'r') as f:
    X = f['value'][:]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_

import numpy as np
cumulative_variance = np.cumsum(explained_variance_ratio)

# Find number of components for 95% and 99% variance
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
n_components_99 = np.argmax(cumulative_variance >= 0.99) + 1

import matplotlib.pyplot as plt
# Plot cumulative explained variance
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, linestyle='-')

# Add horizontal lines at 95% and 99%
plt.hlines(y=0.95, xmin=0, xmax=n_components_95, color='r', linestyle='--', label="95% Variance")
plt.hlines(y=0.99, xmin=0, xmax=n_components_99, color='g', linestyle='--', label="99% Variance")

# Add vertical lines where they intersect
plt.vlines(x=n_components_95, ymin=0, ymax=cumulative_variance[n_components_95-1], color='r', linestyle='--')
plt.vlines(x=n_components_99, ymin=0, ymax=cumulative_variance[n_components_99-1], color='g', linestyle='--')

# Labels and title
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig('cumvar.png')

X_reduced = pca.transform(X_scaled)[:, :n_components_99]
print(f'From {X.shape[1]} features to {X_reduced.shape[1]}')

with h5py.File('output.hdf5', 'r') as f:
    with h5py.File('output_pca.hdf5', 'w') as fd:
        fd['value'] = X_reduced
        
        for k in fd.keys():
            if k != 'value':
                fd[k] = f[k][:]

