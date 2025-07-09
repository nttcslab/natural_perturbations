from collections.abc import Callable
import math
import torch
from torch import Tensor
from omegaconf import DictConfig
from . import eval_output, chip, nn_vmap, equalization


def generate_data(len_symbol: int, len_blank: int, nSamples: int, device) -> tuple[Tensor, Tensor]:
    seq = torch.randint(low=1, high=9, size=(nSamples, len_symbol), dtype=torch.long, device=device)
    zeros1 = torch.zeros((nSamples, len_blank), dtype=torch.long, device=device)
    zeros2 = torch.zeros((nSamples, len_symbol-1), dtype=torch.long, device=device)
    zeros3 = torch.zeros((nSamples, len_symbol+len_blank), dtype=torch.long, device=device)
    marker = 9 * torch.ones((nSamples, 1), dtype=torch.long, device=device)
    x = torch.cat((seq, zeros1, marker, zeros2), dim=1)
    y = torch.cat((zeros3, seq), dim=1)
    return x, y


class CopyingRNN(torch.nn.Module):
    init_state: Tensor
    def __init__(self, cfg: DictConfig, deficient: chip.Deficient):
        super().__init__()
        self.register_buffer('init_state', torch.zeros((1, cfg.num_features), dtype=torch.cfloat), persistent=True)
        self.hidden_size = cfg.num_features
        num_layers = cfg.num_features
        self.rnncell = RNNCellClements(cfg, deficient)
        self.output_clements = chip.ClementsMZI(cfg.num_features, num_layers, deficient)
        # self.softmax = nn_vmap.Softmax(dim=-1)
        num_outputs = 10
        assert num_outputs <= cfg.num_features
        out_start = int((cfg.num_features - num_outputs)/2)
        self.selected_out = torch.arange(out_start, out_start + num_outputs)
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.criterion: Callable[[Tensor, Tensor], Tensor] = eval_output.loss_multinomial
        self.accuracy: Callable[[Tensor, Tensor], float] = eval_output.accuracy
        self.calc_output_fisher: Callable[[Tensor], Tensor] = eval_output.fisher_multinomial_RNN

    def forward(self, inputs: Tensor) -> Tensor:
        nVariation, batch_size, sequence_length = inputs.shape
        one_hot = torch.zeros((nVariation, batch_size, sequence_length, self.hidden_size), device=inputs.device)
        one_hot.scatter_(-1, inputs[:,:,:,None], 1)
        state = self.init_state.expand(batch_size, -1)
        state = torch.stack((state, ) * nVariation)
        output_list = []
        for oh in torch.unbind(one_hot.to(torch.complex64), dim=2):
            state = self.rnncell(oh, state)
            output = self.output_clements(state)
            output = output[..., self.selected_out]
            output_real = output.abs()**2
            # output_real = self.softmax(output_real)
            output_list.append(output_real)
        return torch.stack(output_list, dim=-1)


class RNNCellClements(torch.nn.Module):
    def __init__(self, cfg: DictConfig, deficient: chip.Deficient):
        super().__init__()
        num_layers = cfg.num_features
        self.input = equalization.ClementsMZI_PS(cfg.num_features, num_layers, deficient)
        self.hidden = equalization.ClementsMZI_PS(cfg.num_features, num_layers, deficient)
        self.modReLU = chip.CmodReLU(cfg.num_features)
        torch.nn.init.uniform_(self.input.clements.params, -math.pi, math.pi)
        torch.nn.init.uniform_(self.hidden.clements.params, -math.pi, math.pi)

    def forward(self, inputs: Tensor, state: Tensor) -> Tensor:
        state = self.hidden(state)
        inputs = self.input(inputs)
        return self.modReLU(inputs + state)
