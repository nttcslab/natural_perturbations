import numpy as np
import torch
from torch import Tensor
from torch.func import vmap


def rand_uniform(shape: tuple, scale: float) -> Tensor:
    randu = torch.rand(shape)
    randu -= 0.5
    randu *= 2 * scale
    return randu


class Linear(torch.nn.Module):
    bias: Tensor | None
    def __init__(self, in_features: int, out_features: int, bias: bool=True) -> None:
        super().__init__()
        k = 1 / in_features
        self.weight = torch.nn.Parameter(rand_uniform((1, out_features, in_features), np.sqrt(k)))
        if bias is True:
            self.bias = torch.nn.Parameter(rand_uniform((1, out_features), np.sqrt(k)))
            self.linear = vmap(torch.nn.functional.linear, in_dims=(0, 0, 0))
        else:
            self.bias = None
            self.linear = vmap(torch.nn.functional.linear, in_dims=(0, 0))

    def forward(self, input: Tensor) -> Tensor:
        if self.bias is None:
            return self.linear(input, self.weight)
        else:
            return self.linear(input, self.weight, self.bias)


class ReLU(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.relu = vmap(torch.nn.functional.relu, in_dims=(0,))

    def forward(self, input: Tensor) -> Tensor:
        return self.relu(input)


class Conv2d(torch.nn.Module):
    bias: Tensor | None
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int=1, padding: int=0, bias: bool=True) -> None:
        super().__init__()
        k = 1 / in_channels / kernel_size**2
        self.params = torch.nn.Parameter(rand_uniform((1, out_channels, in_channels, kernel_size, kernel_size), np.sqrt(k)))
        if bias is True:
            self.bias = torch.nn.Parameter(rand_uniform((1, out_channels), np.sqrt(k)))
        else:
            self.bias = None
        self.stride = stride
        self.padding = padding

        def my_conv2d(input: Tensor, params: Tensor, bias: Tensor) -> Tensor:
            return torch.nn.functional.conv2d(input, params, bias=bias, stride=self.stride, padding=self.padding)

        if bias is True:
            self.conv2d = vmap(my_conv2d, in_dims=(0, 0, 0))
        else:
            self.conv2d = vmap(my_conv2d, in_dims=(0, 0, None))

    def forward(self, input: Tensor) -> Tensor:
        return self.conv2d(input, self.params, self.bias)


class MaxPool2d(torch.nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.maxpool2d = vmap(torch.nn.functional.max_pool2d, in_dims=(0, None))
        self.kernel_size = kernel_size

    def forward(self, input: Tensor) -> Tensor:
        return self.maxpool2d(input, self.kernel_size)


class LayerNorm(torch.nn.Module):
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(1, num_features))
        self.bias = torch.nn.Parameter(torch.zeros(1, num_features))

        def my_ln(input: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
            return torch.nn.functional.layer_norm(input, (num_features,), weight, bias)

        self.ln = vmap(my_ln, in_dims=(0, 0, 0))

    def forward(self, input: Tensor) -> Tensor:
        return self.ln(input, self.weight, self.bias)


class Softmax(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.softmax = vmap(torch.nn.functional.softmax, in_dims=(0, None))
    
    def forward(self, input: Tensor) -> Tensor:
        return self.softmax(input, self.dim)
