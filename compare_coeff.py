from dataset import UEDDIEDataset
from torch.linalg import norm

dataset = UEDDIEDataset()

def compare_coeff(name1: str, name2: str):
    coeffs = []
    es = []
    cs = []
    for x, e, c, y, name in [dataset.get(x, return_name=True) for x in range(len(dataset))]:
        if name == name1 or name == name2:
            coeffs.append(x)
            es.append(e)
            cs.append(c)
    
    assert len(coeffs) == 2
    
    diff = coeffs[0] - coeffs[1]
    print(diff)
    print(f'Norm of coefficient diff: {norm(diff)}')
    print(es[0] - es[1])
    print(cs[0] - cs[1])

compare_coeff('C0491_A0090-d3_91', 'C0491_A0090-d4_74')
compare_coeff('C0491_A0090-d5_56', 'C0491_A0090-d4_74')
