from collections.abc import Callable
import torch
from torch import Tensor
from . import eval_output, chip


def generate_data(num_features: int, nSamples_train: int, nSamples_test: int, device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    matrix = torch.randn((num_features, num_features), dtype=torch.cfloat, device=device)
    train_data = torch.randn((nSamples_train, num_features), dtype=torch.cfloat, device=device)
    train_targets = torch.einsum('mn,bn->bm', matrix, train_data).abs()
    test_data = torch.randn((nSamples_test, num_features), dtype=torch.cfloat, device=device)
    test_targets = torch.einsum('mn,bn->bm', matrix, test_data).abs()
    return train_data, train_targets, test_data, test_targets


class Equalization(torch.nn.Module):
    def __init__(self, nFeatures: int, deficient: chip.Deficient) -> None:
        super().__init__()
        self.module = SVD_Clements_Diag_Clements(nFeatures, 1.0, deficient)
        self.criterion = torch.nn.MSELoss()
        self.accuracy: Callable[[Tensor, Tensor], float] = eval_output.r2_score
        self.calc_output_fisher: Callable[[Tensor], Tensor] = eval_output.fisher_normal

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x).abs()


class SVD_Clements_Diag_Clements(torch.nn.Module):
    def __init__(self, nFeatures: int, init_diag_scale: float, deficient: chip.Deficient) -> None:
        super().__init__()
        self.module_list = torch.nn.ModuleList([
            ClementsMZI_PS(nFeatures, nFeatures, deficient),
            chip.Diag(nFeatures, init_diag_scale),
            ClementsMZI_PS(nFeatures, nFeatures, deficient)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        x = x.to(dtype=torch.cfloat)
        for mod in self.module_list:
            x = mod(x)
        return x


class ClementsMZI_PS(torch.nn.Module):
    def __init__(self, nFeatures: int, nLayers: int, deficient: chip.Deficient) -> None:
        super().__init__()
        self.nFeatures = nFeatures
        self.clements = chip.ClementsMZI(nFeatures, nLayers, deficient)
        self.ps = chip.ModulePS(nFeatures, deficient)

    def forward(self, input: Tensor) -> Tensor:
        output = self.clements(input)
        return self.ps(output)
