from pathlib import Path

energies_csv = Path('data/Database_SRSMP2.csv')
energies_dat = Path('energies.dat')

# Parse energies csv
with open(energies_csv) as fd: 
    lines = fd.readlines()[1:]
    lines = [line.split(',') for line in lines]
    csv_systems = {system.strip(): float(energy) / 4.184 for energy, system in lines}

# Parse dat energies
with open(energies_dat) as fd:
    lines = fd.readlines()
    lines = [line.split(' ') for line in lines]
    dat_systems = {system[:-4]: 627.509 * float(energy) for system, energy in lines}

# Restrict csv_systems and dat_systems to systems in common
systems = set(csv_systems.keys()).intersection(set(dat_systems.keys()))
csv_systems = {system: value for system, value in csv_systems.items() if system in systems}
dat_systems = {system: value for system, value in dat_systems.items() if system in systems} 

diffs = {}
for system in csv_systems:
    diffs[system] = dat_systems[system] - csv_systems[system]

# Sort diffs ascending
diffs = {k: v for k, v in sorted(diffs.items(), key=lambda item: item[1])}

print('Top 10 systems with greatest diffs:')
for key, value in list(diffs.items())[::-1][:10]:
    print(f'{key}: {value}')

print('Top 10 systems with lowest diffs:')
for key, value in list(diffs.items())[:10]:
    print(f'{key}: {value}')

# Create violin plot to compare
import pandas as pd
import numpy as np
x = ['Ref. SRS-MP2', 'Dual-basis SRS-MP2']
y = [list(csv_systems.values()), list(dat_systems.values())]
df = pd.DataFrame({
    'values': np.concatenate([np.array(y[0]), np.array(y[1])]),
    'group': ['Ref. SRS-MP2'] * len(systems) + ['Dual-basis SRS-MP2'] * len(systems)
})

import matplotlib.pyplot as plt
import seaborn as sns
sns.violinplot(data=df, x='group', y='values', orient='v')
plt.xlabel('Group')
plt.ylabel('Interaction Energy [kcal/mol]')
plt.savefig('compare_energies_violin.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot residuals
plt.scatter(np.arange(1, len(diffs)+1), np.array(list(diffs.values())), color='green')
plt.axhline(y=0, color='black', linestyle='--')
plt.axhline(y=1, color='red', linestyle='--')
plt.axhline(y=-1, color='red', linestyle='--')
plt.xlabel('')
plt.ylabel('Residual [kcal/mol]')
plt.savefig('compare_energies_residuals.png', dpi=300, bbox_inches='tight')
