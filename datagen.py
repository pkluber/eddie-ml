from pathlib import Path
import tarfile
import os

from density import cube_utils

import argparse

parser = argparse.ArgumentParser(description='Generate deformation densities (as .cube files).')
parser.add_argument('--resolution', type=float, default=0.5, help='Resolution of grid')
parser.add_argument('--extension', type=float, default=5.0, help='Extension of grid')
parser.add_argument('--level', type=int, default=1, help='Level of LDA grid')

args = parser.parse_args()

neutral_tar = 'data/neutral-dimers.tar.gz' 
charged_tar = 'data/charged-dimers.tar.gz'

METHOD = 'LDA'

def list_contents(tarfile_path: str) -> list[str]:
    outputs = []
    with tarfile.open(tarfile_path, 'r:gz') as tar:
        for member in tar.getmembers():
            filename = os.path.basename(member.name)
            outputs.append(filename)

    return outputs

neutral_list = list_contents(neutral_tar)
charged_list = list_contents(charged_tar)

data_dir = Path('data/bcurves')
for file in data_dir.rglob('*'):
    if file.is_file() and file.suffix == '.xyz':
        # Check if .cube file already exists
        cube_path = file.parent / f'{file.name[:-4]}.cube'
        if cube_path.exists():
            print(f'Found .cube file for {file.name}', flush=True)
            continue

        if file.name in neutral_list:
            charges = [0, 0, 0]
        elif file.name.startswith('C'):
            charges = [1, -1, 0]
        else:
            continue

        print(f'Processing {file} with charges {charges}', flush=True)
        cube_utils.dimer_cube_difference(str(file), METHOD, resolution=args.resolution, extension=args.extension, level=args.level, charges=charges, write_cube=True, path=str(file.parent))
        print('', flush=True)

