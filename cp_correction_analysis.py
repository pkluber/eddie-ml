def parse_energy_dat(filename: str):
    with open(filename) as fd:
        lines = fd.readlines()
        lines = [line.split(' ') for line in lines]
        energies = {system.strip(): 627.509 * float(value) for system, value in lines}
        return energies

base_energies = parse_energy_dat('energies.dat')
cp_energies = parse_energy_dat('cp_energies.dat')

diffs = {}
for system in cp_energies:
    diffs[system] = cp_energies[system] - base_energies[system]

# Create violin plot of residuals
import pandas as pd
import numpy as np
df = pd.DataFrame({
    'values': np.array(diffs.values()),
    'group': [''] * len(diffs)
})

import matplotlib.pyplot as plt
import seaborn as sns
sns.violinplot(data=df, y='group', x='values', orient='h')
plt.xlabel('CP Corrections (kcal/mol)')
plt.ylabel('')
plt.savefig('cp_corrections.png', dpi=300, bbox_inches='tight')
plt.close()
    
