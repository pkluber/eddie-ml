from typing import Tuple

POS_CHARGED_AMINOS = ['ARG', 'LYS']
NEG_CHARGED_AMINOS = ['ASP', 'GLU']

def get_amino_charge(amino: str) -> int:
    if amino in POS_CHARGED_AMINOS:
        return 1
    elif amino in NEG_CHARGED_AMINOS:
        return -1
    else:
        return 0

def get_charges(filename: str) -> Tuple[int, int, int]:
    if filename.startswith('S66'):
        return 0, 0, 0
    elif filename.startswith('SSI'):
        split = filename.split('-')
        aa1 = split[1][3:]
        aa2 = split[2][3:]
        charge1 = get_amino_charge(aa1)
        charge2 = get_amino_charge(aa2)
        if charge1 == charge2 or charge1 + charge2 != 0: 
            return 0, 0, 0
        else:
            return 0, charge1, charge2
    elif filename.startswith('C_'): # IL174
        return 0, 1, -1
    elif filename.startswith('C'): # extraILs
        if filename.startswith('C0491_A0090') or filename.startswith('C2004_A0073'):
            return 0, -1, 1
        else:
            return 0, 1, -1
