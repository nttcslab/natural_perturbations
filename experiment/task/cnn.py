from collections.abc import Callable
import torch
from torch import Tensor
from . import eval_output, nn_vmap


class CNN_MNIST(torch.nn.Module):
    def __init__(self, inner_ch: int, linear_width: int):
        super().__init__()
        num_ch = 1
        wh = 28
        kernel_size = 3
        km1 = kernel_size - 1
        self.last_wh = int((int((wh - km1) / 2) - km1) / 2) - km1
        self.cnn = torch.nn.ModuleList([
            nn_vmap.Conv2d(num_ch, inner_ch, kernel_size),
            nn_vmap.ReLU(),
            nn_vmap.MaxPool2d(2),
            nn_vmap.Conv2d(inner_ch, inner_ch, kernel_size),
            nn_vmap.ReLU(),
            nn_vmap.MaxPool2d(2),
            nn_vmap.Conv2d(inner_ch, inner_ch, kernel_size),
            nn_vmap.ReLU()
        ])
        self.fully = torch.nn.ModuleList([
            nn_vmap.Linear((self.last_wh**2)*inner_ch, linear_width),
            nn_vmap.ReLU(),
            nn_vmap.Linear(linear_width, 10),
            # nn_vmap.Softmax(dim=-1)
        ])
        self.criterion = torch.nn.CrossEntropyLoss()
        # self.criterion: Callable[[Tensor, Tensor], Tensor] = eval_output.loss_multinomial
        self.accuracy: Callable[[Tensor, Tensor], float] = eval_output.accuracy
        self.calc_output_fisher: Callable[[Tensor], Tensor] = eval_output.fisher_multinomial

    def forward(self, x: Tensor) -> Tensor:
        for f in self.cnn:
            x = f(x)
        xsh = x.shape
        x = x.view(*xsh[:-3], -1)
        for f in self.fully:
            x = f(x)
        return x
