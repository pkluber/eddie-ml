import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FunnelMLP(nn.Module):
    def __init__(self, in_dim: int, h: int = 128, depth: int = 10):
        super().__init__()

        # Create a funnel {depth} long
        layers = [nn.Linear(in_dim, h)]
        for x in range(depth):
            layers.append(nn.Softplus())
            
            h_in = int(round(h / depth * (depth - x)))
            h_out = int(round(h / depth * (depth - x - 1))) if x != depth - 1 else 1

            layers.append(nn.Linear(h_in, h_out))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class UEDDIENetwork(nn.Module):
    def __init__(self, X_shape):
        super().__init__()
        self.elem_subnets = nn.ModuleDict({str(e): FunnelMLP(X_shape[2]) for e in range(4)})
        self.charge_subnets = nn.ModuleDict({str(c): FunnelMLP(X_shape[2]) for c in range(-1, 2)})

    def forward(self, X: torch.Tensor, E: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        per_atom_IE = torch.zeros(X.shape[:-1], device=X.device)
        for b in range(E.shape[0]):
            for x in range(E.shape[1]):
                e = int(E[b, x].item())
                if str(e) not in self.elem_subnets:
                    continue
                per_atom_IE[b, x] = self.elem_subnets[str(e)](X[b,x,...])

        per_atom_CF = torch.zeros(X.shape[:-1], device=X.device)
        for b in range(C.shape[0]):
            for x in range(C.shape[1]):
                c = C[b, x].item()
                if str(c) not in self.charge_subnets:
                    continue

                per_atom_CF = self.charge_subnets[str(c)](X[b,x,...])

        return -per_atom_IE.sum(dim=1) * per_atom_CF.sum(dim=1)


# Example usage
if __name__ == '__main__':
    X = torch.randn(16, 34, 17)
    E = torch.randn(16, 34)
    Y = torch.randn(16)
    model = UEDDIENetwork(X.shape)
    y = model(X, E)
    print(f'Input shape: {X.shape}, {E.shape}\nOutput shape: {y.shape}\nExpected shape: {Y.shape}')
