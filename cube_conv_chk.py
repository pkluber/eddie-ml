from pathlib import Path

any_failed = False

data_dir = Path('data/bcurves')
for file in data_dir.rglob('*'):
    if file.is_file() and file.suffix == '.xyz':
        filename = file.name[:-4]
        if not (file.parent / f'{filename}.cube').exists():
            print(f'{filename} failed to converge!')
            any_failed = True

if not any_failed:
    print('All systems have converged .cube files!')
