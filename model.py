import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def tag(layers: OrderedDict, id_num: int) -> OrderedDict:
    tagged = OrderedDict([])
    for label, layer in layers.items():
        tagged[label + str(id_num)] = layer
    return tagged


def linear_block_factory() -> Callable:
    id_num = 0

    def linear_block(in_dim, out_dim, dropout=0.5):
        layers = OrderedDict([
            ('dense', nn.Linear(in_dim, out_dim)),
            ('act', nn.Softplus()),
            ('norm', nn.LayerNorm(out_dim)),
            ('dropout', nn.Dropout(dropout))
        ])
        nonlocal id_num
        tagged_layers = tag(layers, id_num)
        id_num += 1

        return nn.Sequential(tagged_layers)

    return linear_block

class UEDDIENetwork(nn.Module):
    pass
