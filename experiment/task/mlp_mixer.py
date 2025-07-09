from collections.abc import Callable
import torch
from torch import Tensor
from einops.layers.torch import Rearrange
from . import eval_output, nn_vmap


class MLPmixer(torch.nn.Module):
    def __init__(self, nMixer: int, nChannel: int, kernel_size: int, in_channels: int, image_size: int) -> None:
        super().__init__()
        assert image_size % kernel_size == 0, 'Cannot divide image evenly.'
        nPatch = (image_size // kernel_size) ** 2
        self.mixer_list = torch.nn.ModuleList(
            [Mixer(nPatch, nChannel) for i in range(nMixer)]
        )
        self.preprocess = torch.nn.Sequential(
            nn_vmap.Conv2d(in_channels, nChannel, kernel_size, kernel_size, bias=True),
            Rearrange('v b c h w -> v b (h w) c')
        )
        self.layer_norm = nn_vmap.LayerNorm(nChannel)
        self.outLinear = nn_vmap.Linear(nChannel, 10, bias=True)
        # self.softmax = nn_vmap.Softmax(dim=-1)
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.criterion: Callable[[Tensor, Tensor], Tensor] = eval_output.loss_multinomial
        self.accuracy: Callable[[Tensor, Tensor], float] = eval_output.accuracy
        self.calc_output_fisher: Callable[[Tensor], Tensor] = eval_output.fisher_multinomial
        print(f'MLPmixer: nPatch={nPatch}, nChannel={nChannel}, nMixer={nMixer}')

    def forward(self, x: Tensor) -> Tensor:
        x = self.preprocess(x)
        for mixer in self.mixer_list:
            x = mixer(x)
        x = self.layer_norm(x)
        x = x.mean(dim=-2)
        x = self.outLinear(x)
        # x = self.softmax(x)
        return x


class Mixer(torch.nn.Module):
    def __init__(self, nPatch: int, nChannel: int) -> None:
        super().__init__()
        self.patch_mix = torch.nn.Sequential(
            nn_vmap.LayerNorm(nChannel),
            Rearrange('v b p c -> v b c p'),
            MLP(nPatch),
            Rearrange('v b c p -> v b p c')
        )
        self.channel_mix = torch.nn.Sequential(
            nn_vmap.LayerNorm(nChannel),
            MLP(nChannel)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_mix(x) + x
        return self.channel_mix(x) + x


class MLP(torch.nn.Module):
    def __init__(self, nFeature: int) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            nn_vmap.Linear(nFeature, nFeature, bias=True),
            nn_vmap.ReLU(),
            nn_vmap.Linear(nFeature, nFeature, bias=True)
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
