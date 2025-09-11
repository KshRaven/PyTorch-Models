
import torch
import torch.nn as nn

from torch import Tensor


class Transpose(nn.Module):
    def __init__(self, dim0=-1, dim1=-2):
        super(Transpose, self).__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, tensor: Tensor):
        return torch.transpose(tensor, self.dim0, self.dim1)

    def extra_repr(self) -> str:
        return f"dim0={self.dim0}, dim1={self.dim1}"


class Ignore(nn.Module):
    def __init__(self, rate: float = 0):
        super(Ignore, self).__init__()
        assert 0 <= rate < 1
        self.rate = rate

    def forward(self, tensor: Tensor):
        if self.training and torch.rand(1).item() < self.rate:
            tensor = tensor * 0
        return tensor

    def extra_repr(self) -> str:
        return f"rate={self.rate}"
