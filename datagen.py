from pathlib import Path
import tarfile
import os

from density import cube_utils

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
        cube_utils.dimer_cube_difference(str(file), METHOD, resolution=0.5, extension=5, charges=charges, write_cube=True, path=str(file.parent))
        print('', flush=True)

