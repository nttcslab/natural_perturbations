from sklearn import metrics
import torch
from torch import Tensor


# def loss_multinomial(input: Tensor, target: Tensor) -> Tensor:
#     log_prob = torch.log(input)
#     return torch.nn.functional.nll_loss(log_prob, target)


def accuracy(output: Tensor, label: Tensor) -> float:
    return torch.eq(torch.argmax(output.detach(), dim=1), label).float().mean().item()


def r2_score(output: Tensor, target: Tensor) -> float:
    r2score = metrics.r2_score(target.detach().cpu().numpy(), output.detach().cpu().numpy())
    return float(r2score)   


def fisher_multinomial(logits: Tensor) -> Tensor:
    probability = torch.softmax(logits.detach(), dim=1)
    num_classes = probability.shape[1]
    log_derivative = (-probability[None]).repeat(num_classes, 1, 1)
    for m in range(num_classes):
        log_derivative[m, :, m] += 1
    return torch.einsum('bo,obm,obn->bmn', probability, log_derivative, log_derivative)


def fisher_multinomial_RNN(logits: Tensor) -> Tensor:
    probability = torch.softmax(logits.detach(), dim=1)
    num_classes = probability.shape[1]
    log_derivative = (-probability[None]).repeat(num_classes, 1, 1, 1)
    for m in range(num_classes):
        log_derivative[m, :, m] += 1
    return torch.einsum('bos,obms,obns->bmns', probability, log_derivative, log_derivative)


def fisher_normal(output: Tensor) -> Tensor:
    num_sample, dim_output = output.shape
    eye = torch.eye(dim_output, device=output.device, dtype=output.dtype)
    return eye[None].repeat(num_sample, 1, 1)
