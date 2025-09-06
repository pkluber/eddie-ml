import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple

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
        return self.net(x).squeeze(-1)

def create_mask(X: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = mask.unsqueeze(-1)
    mask = mask.tile((1,1,X.shape[2]))
    X_masked = torch.where(mask, X, 0)
    return mask, X_masked


class UEDDIENetwork(nn.Module):
    def __init__(self, X_shape):
        super().__init__()
        self.elem_subnets = nn.ModuleDict({str(e): FunnelMLP(X_shape[2]) for e in range(4)})
        self.charge_subnets = nn.ModuleDict({str(c): FunnelMLP(X_shape[2]) for c in range(-1, 2)})

    def forward(self, X: torch.Tensor, E: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Sanitize E, C, make sure they're int
        E = E.to(torch.int64)
        C = C.to(torch.int64)

        # Atomistic approximation
        per_atom_IE = torch.zeros(X.shape[:-1], device=X.device)

        # Apply elementwise subnets
        for e in self.elem_subnets.keys():
            mask, X_masked = create_mask(X, E == int(e))
            per_atom_IE = torch.where(mask[:, :, 0], self.elem_subnets[e](X_masked), per_atom_IE)

        # Apply charge scaling subnets
        for c in self.charge_subnets.keys():
            mask, X_masked = create_mask(X, C == int(c))
            per_atom_IE = torch.where(mask[:, :, 0], per_atom_IE * (1 + self.charge_subnets[c](X_masked)), per_atom_IE)

        return -per_atom_IE.sum(dim=1)


# Example usage
if __name__ == '__main__':
    X = torch.randn(16, 34, 17)
    E = torch.randn(16, 34)
    C = torch.randn(16, 34)
    Y = torch.randn(16)
    model = UEDDIENetwork(X.shape)
    y = model(X, E, C)
    print(f'Input shape: {X.shape}, {E.shape}, {C.shape}\nOutput shape: {y.shape}\nExpected shape: {Y.shape}')
