import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import itertools

from typing import Tuple

def create_funnel(h: int, depth: int):
    dims = []
    for x in range(depth):
        h_in = int(round(h / depth * (depth - x)))
        h_out = int(round(h / depth * (depth - x - 1))) if x != depth - 1 else 1
        dims.append((h_in, h_out))

    return dims

def create_wide_funnel(h: int, depth: int):
    dims = []

    linspace = np.linspace(h, h/2, depth)
    linspace = linspace.astype(int)
    for x in range(depth):
        h_in = linspace[x]
        if x == depth - 1:
            h_out = 1
        else:
            h_out = linspace[x+1]

        dims.append((h_in, h_out))

    return dims

class FunnelMLP(nn.Module):
    def __init__(self, in_dim: int, h: int = 128, depth: int = 10):
        super().__init__()

        # Create a funnel {depth} long
        layers = [nn.Linear(in_dim, h)]
        funnel = create_wide_funnel(h, depth)
        for h_in, h_out in funnel:
            layers.append(nn.Softplus())
            layers.append(nn.Linear(h_in, h_out))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

def create_mask(X: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = mask.unsqueeze(-1)
    mask = mask.tile((1,1,X.shape[2]))
    X_masked = torch.where(mask, X, 0)
    return mask, X_masked


class UEDDIEPrototype(nn.Module):
    def __init__(self, X_shape: tuple, elem_depth: int = 10, charge_depth: int = 10):
        super().__init__()
        self.elem_subnets = nn.ModuleDict({str(e): FunnelMLP(X_shape[2], depth=elem_depth) for e in range(4)})
        self.charge_subnets = nn.ModuleDict({str(c): FunnelMLP(X_shape[2], depth=charge_depth) for c in range(-1, 2)})

    def forward(self, X: torch.Tensor, E: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Sanitize E, C, make sure they're int
        E = E.to(torch.int64)
        C = C.to(torch.int64)

        # Atomistic approximation
        per_atom_IE = torch.zeros(X.shape[:-1], device=X.device)

        # Apply element subnets
        for e in self.elem_subnets.keys():
            mask, X_masked = create_mask(X, E == int(e))
            per_atom_IE = torch.where(mask[:, :, 0], self.elem_subnets[e](X_masked), per_atom_IE)

        # Apply charge scaling subnets
        for c in self.charge_subnets.keys():
            mask, X_masked = create_mask(X, C == int(c))
            per_atom_IE = torch.where(mask[:, :, 0], per_atom_IE * torch.exp(self.charge_subnets[c](X_masked)), per_atom_IE)

        return -per_atom_IE.sum(dim=1)

class UEDDIEMoE(nn.Module):
    def __init__(self, X_shape: tuple, num_experts: int = 8):
        super().__init__()
        self.experts = nn.ModuleList([UEDDIENetwork(X_shape) for _ in range(num_experts)])
        self.gating = nn.Sequential(
            nn.Linear(X_shape[-1], num_experts), 
            nn.Softmax(dim=-1)
        )

    def forward(self, X: torch.Tensor, E: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Apply atomistic decomposition to gating
        gate_outputs = self.gating(X).sum(dim=1)

        expert_outputs = torch.stack([expert(X, E, C) for expert in self.experts], dim=-1)

        return (gate_outputs * expert_outputs).sum(dim=-1)


class SelfAttention(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, device: torch.device, embed_dim: int, ff_hidden_dim: int):
        super().__init__()
        self.attn = SelfAttention(embed_dim)
        self.attn.to(device)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.Softplus(),
            nn.Linear(ff_hidden_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Self-attention + residual + norm
        attn_out = self.attn(x)
        x = self.norm1(x + attn_out)

        # Feed-forward + residual + norm
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        return x

class FinetunerSubnet(nn.Module):
    def __init__(self, device: torch.device, X_shape: tuple, depth: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Sequential(*[TransformerBlock(device, X_shape[-1], 128) for _ in range(depth)]),
            nn.Linear(X_shape[-1], 1),
        )
        self.net.to(device)
 
    def forward(self, X: torch.Tensor):
        return self.net(X).squeeze(-1) / 1000

class UEDDIEFinetuner(nn.Module):
    def __init__(self, device: torch.device, X_shape: tuple, subnet_depth: int = 2):
        super().__init__()
        self.subnets = nn.ModuleDict(
            {f'{str(e)},{str(c)}': FinetunerSubnet(device, X_shape, depth=subnet_depth)
             for e, c in itertools.product(range(4), range(-1, 2))}
        )
        self.subnets.to(device)

    def forward(self, X: torch.Tensor, E: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Sanitize E, C
        E = E.to(torch.int64)
        C = C.to(torch.int64)

        per_atom_finetune = torch.zeros(X.shape[:-1], device=X.device)

        # Apply subnets
        for e in range(4):
            for c in range(-1, 2):
                mask, X_masked = create_mask(X, torch.logical_and((E == e), (C == c)))
                per_atom_finetune = torch.where(mask[:, :, 0], self.subnets[f'{str(e)},{str(c)}'](X_masked), per_atom_finetune)

        return -per_atom_finetune.sum(dim=1)

# Swish-gated GLU or SwiGLU implementation
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()

        self.fc1 = nn.Linear(d_model, 2*d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x_a, x_b = x.chunk(2, dim=-1)
        x = x_a * F.silu(x_b)
        return self.fc2(x)

# Multi-head self-attention implementation
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        
        # Round up head dimension 
        self.d_padded = math.ceil(d_model / num_heads) * num_heads
        self.d_head = self.d_padded // num_heads
        
        # QKV projections with padded dimension
        self.qkv = nn.Linear(d_model, 3*self.d_padded)
        self.proj = nn.Linear(self.d_padded, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape

        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to [B, num_heads, N, d_head]
        q = q.view(B, N, self.num_heads, self.d_head).transpose(1,2)
        k = k.view(B, N, self.num_heads, self.d_head).transpose(1,2)
        v = v.view(B, N, self.num_heads, self.d_head).transpose(1,2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn_probs = F.softmax(attn_scores, dim=-1) # [B, num_heads, N, N]
        out = torch.matmul(attn_probs, v)  # [B, num_heads, N, d_head]

        # Combine heads back, make sure contiguous again after all the transposes
        out = out.transpose(1,2).contiguous().view(B, N, self.d_padded)
        
        # Final step: project from d_padded to d_model 
        return self.proj(out)

class UEDDIETransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = SwiGLU(d_model, d_ff)

    def forward(self, x):
        # Pre-norm then apply MHSA
        x = self.norm1(x)
        attn_out = self.attn(x)
        
        # Residual connection to another pre-norm then apply SwiGLU 
        x = self.norm2(x + attn_out)
        ff_out = self.ff(x)
        
        # Return residual of SwiGLU and pre-normed MHSA
        return x + ff_out

# Subnet of transformer blocks that projects to 1 dim at the end 
class UEDDIESubnet(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, depth: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Sequential(*[UEDDIETransformerBlock(d_model, num_heads, d_ff) for _ in range(depth)]),
            nn.Linear(d_model, 1)
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X).squeeze(-1)


class UEDDIENetwork(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, d_ff: int = 128, depth_e: int = 10, depth_c: int = 10):
        super().__init__()
        self.elem_subnets = nn.ModuleDict({str(e): UEDDIESubnet(d_model, num_heads, d_ff, depth_e) for e in range(4)})
        self.charge_subnets = nn.ModuleDict({str(c): UEDDIESubnet(d_model, num_heads, d_ff, depth_c) for c in range(-1, 2)})

    def forward(self, X: torch.Tensor, E: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        # Sanitize E, C, make sure they're int
        E = E.to(torch.int64)
        C = C.to(torch.int64)

        # Atomistic approximation
        per_atom_IE = torch.zeros(X.shape[:-1], device=X.device)

        # Apply element subnets
        for e in self.elem_subnets.keys():
            mask, X_masked = create_mask(X, E == int(e))
            per_atom_IE = torch.where(mask[:, :, 0], self.elem_subnets[e](X_masked), per_atom_IE)

        # Apply charge scaling subnets
        for c in self.charge_subnets.keys():
            mask, X_masked = create_mask(X, C == int(c))
            per_atom_IE = torch.where(mask[:, :, 0], per_atom_IE * torch.exp(self.charge_subnets[c](X_masked)), per_atom_IE)

        return -per_atom_IE.sum(dim=1)

# Example usage
if __name__ == '__main__':
    X = torch.randn(16, 34, 17)
    E = torch.randn(16, 34)
    C = torch.randn(16, 34)
    Y = torch.randn(16)
    model = UEDDIENetwork(X.shape[-1])
    y = model(X, E, C)
    print(f'Input shape: {X.shape}, {E.shape}, {C.shape}\nOutput shape: {y.shape}\nExpected shape: {Y.shape}')

    print(f'UEDDIENetwork has {sum(param.numel() for param in model.parameters())} parameters')

    moe_model = UEDDIEMoE(X.shape)
    y = moe_model(X, E, C)
    print(f'MoE output shape: {y.shape}')

    finetuner = UEDDIEFinetuner(torch.device('cpu'), X.shape)
    y = finetuner(X, E, C)
    print(y)
    print(f'Finetuner output shape: {y.shape}')
    print(f'UEDDIEFinetuner has {sum(param.numel() for param in finetuner.parameters())} parameters')

