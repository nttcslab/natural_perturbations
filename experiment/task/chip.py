import numpy as np
import torch
from torch import Tensor
from dataclasses import dataclass
import mzi_onn_sim


@dataclass
class Deficient:
    split: float = 0
    eps_radius: float = 0
    eps_angle: float = 0

    def set_deficient(self, cfg_deficient, cfg_defmul):
        self.split = cfg_defmul * cfg_deficient.split
        self.eps_angle = cfg_defmul * cfg_deficient.eps_angle
        self.eps_radius = cfg_defmul * cfg_deficient.eps_radius


class ModulePS(torch.nn.Module):
    def __init__(self, nFeatures: int, deficient: Deficient):
        super().__init__()
        self.params = torch.nn.Parameter(torch.zeros(1, nFeatures))
        radius = 1.0 - deficient.eps_radius * torch.rand(nFeatures)
        phi = deficient.eps_angle * (2.0 * torch.rand(nFeatures) - 1)
        self.register_buffer('atten', radius*torch.exp(1.j*phi), persistent=True)

    def forward(self, input: Tensor) -> Tensor:
        return mzi_onn_sim.zo.forwardPS(input, self.params, self.atten)


class ModulePSBS(torch.nn.Module):
    def __init__(self, nFeatures: int, index: Tensor, deficient: Deficient):
        super().__init__()
        self.nFeatures = nFeatures
        nLayers, nAngles, _ = index.shape
        self.params = torch.nn.Parameter(torch.randn(1, nLayers, nAngles))
        self.register_buffer('index', index, persistent=True)
        self.register_buffer('split', torch.randn(nLayers, nAngles, dtype=torch.float)*deficient.split, persistent=True)
        radius = 1.0 - deficient.eps_radius * torch.rand(nLayers, nFeatures)
        phi = deficient.eps_angle * (2.0 * torch.rand(nLayers, nFeatures) - 1)
        self.register_buffer('atten', radius*torch.exp(1.j*phi), persistent=True)

    def forward(self, input: Tensor) -> Tensor:
        assert input.shape[0] <= self.params.shape[0], f'insufficient variations {self.params.shape[0]} of parameters for input variations {input.shape[0]},'
        return mzi_onn_sim.zo.forwardPSBS(input, self.params, self.index, self.split, self.atten)


class ClementsMZI(ModulePSBS):
    def __init__(self, nFeatures: int, nLayers: int, deficient: Deficient):
        self.nLayers = nLayers
        indexA, indexB = _make_indexAB(nFeatures)
        indexAB = ([indexA] * 2 + [indexB] * 2) * int(nLayers/2)
        if nLayers % 2 == 1:
            indexAB += [indexA] * 2
        indexABtensor = torch.tensor(indexAB, dtype=torch.int32)
        super().__init__(nFeatures, indexABtensor, deficient)


class Diag(torch.nn.Module):
    def __init__(self, input_size: int, init_scale: float=1.0):
        super().__init__()
        self.coeff = torch.nn.Parameter(init_scale * torch.ones((1, input_size,), dtype=torch.float))

    def forward(self, input: Tensor) -> Tensor:
        return torch.einsum('vf,vbf->vbf', self.coeff, input)
   

class CmodReLU(torch.nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.zeros((1, input_size), dtype=torch.float))

    def forward(self, input: Tensor) -> Tensor:
        return mzi_onn_sim.zo.forwardmodReLU(input, self.bias)


def _make_indexAB(nFeatures: int) -> tuple[list, list]:
    if nFeatures % 2 == 0:
        nAnglesAB = int(nFeatures/2)
        indexA = [[i*2, i*2+1] for i in range(nAnglesAB)]
        indexB = [[i*2+1, i*2+2] for i in range(nAnglesAB-1)] + [[np.invert(0), np.invert(nFeatures-1)]]
    else:
        nAnglesAB = int(nFeatures/2) + 1
        indexA = [[i*2, i*2+1] for i in range(nAnglesAB-1)] + [[np.invert(nFeatures-1)] * 2]
        indexB = [[i*2+1, i*2+2] for i in range(nAnglesAB-1)] + [[np.invert(0)] * 2]
    return indexA, indexB
