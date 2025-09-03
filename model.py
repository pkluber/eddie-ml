import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ElementMLP(nn.Module):
    def __init__(self, in_dim: int, h: int = 128):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.Softplus(),
            nn.Linear(h, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class UEDDIENetwork(nn.Module):
    def __init__(self, X_shape):
        super().__init__()
        self.subnets = nn.ModuleDict({str(e): ElementMLP(X_shape[2]) for e in range(4)})

    def forward(self, X: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
        per_atom_IE = torch.zeros(X.shape[:-1], device=X.device)
        for b in range(E.shape[0]):
            for x in range(E.shape[1]):
                e = int(E[b, x].item())
                if str(e) not in self.subnets:
                    continue
                per_atom_IE[b, x] = self.subnets[str(e)](X[b,x,...])

        return -per_atom_IE.sum(dim=1)


# Example usage
if __name__ == '__main__':
    X = torch.randn(16, 34, 17)
    E = torch.randn(16, 34)
    Y = torch.randn(16)
    model = UEDDIENetwork(X.shape)
    y = model(X, E)
    print(f'Input shape: {X.shape}, {E.shape}\nOutput shape: {y.shape}\nExpected shape: {Y.shape}')
