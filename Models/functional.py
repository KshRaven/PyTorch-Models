
from Models.util.fancy_text import cmod, Fore

import torch
import torch.nn as nn
import numpy as np
import math

from torch import Tensor
from typing import Union, Iterable


def model_size(model: nn.Module):
    return np.sum([param.numel() * param.element_size() for param in model.parameters()]) / (1024 ** 2)


def model_params(model: nn.Module):
    return np.sum([param.numel() for param in model.parameters()])


COLOURS = {lbl: clr for lbl, clr in vars(Fore).items() if
           not any(fltr in lbl for fltr in ['BLACK', 'WHITE']) and 'LIGHT' in lbl}


class color_fetch:
    def __init__(self):
        self.current_idx = 0
        self.colors = list(COLOURS.values())
        self.color_num = len(self.colors)

    def __call__(self):
        self.current_idx += 1
        return self.colors[self.current_idx % self.color_num]


get = color_fetch()


def get_tensor_info(tensor: Tensor, label: str = None, verbose: int = None, color: Fore = None):
    tensor = tensor.detach().clone().cpu().type(torch.float32)
    label = f"{cmod(label, get() if not color else color)} => {tensor.shape}" if label is not None else ""
    stats = f"\n\tmean={tensor.mean()}, std={tensor.std()}, max={tensor.max()}, min={tensor.min()}\n" if verbose and verbose >= 3 else ""
    details = cmod(f"\n{tensor}\n", color if color else Fore.LIGHTWHITE_EX) if verbose and verbose >= 4 else ""
    return f"{label}{stats}{details}"


# TODO: Fix to work for tuple of stride and dilation
def calc_padding(
        kernel_size: int | tuple[int, ...],
        stride: int | tuple[int, ...] = 1,
        dilation: int | tuple[int, ...] = 1,
):
    if isinstance(kernel_size, int):
        return math.ceil(((kernel_size - 1) * dilation - (stride - 1)) / 2)
    elif isinstance(kernel_size, (tuple, list)):
        return tuple([
            math.ceil(((k - 1) * dilation - (stride - 1)) / 2)
            for k in kernel_size
        ])
    else:
        raise ValueError(f"{kernel_size} is not a valid kernel size")


def display(label: str, var, check=None):
    if not (var if check is None else check):
        return ''
    else:
        return f", {label}={var}"


def get_conv(inputs: Union[Iterable[int], int], reverse=False):
    if isinstance(inputs, (int, float)):
        inputs = [1 for _ in range(int(inputs))]
    if len(inputs) == 1:
        return nn.Conv1d if not reverse else nn.ConvTranspose1d
    elif len(inputs) == 2:
        return nn.Conv2d if not reverse else nn.ConvTranspose2d
    elif len(inputs) == 3:
        return nn.Conv3d if not reverse else nn.ConvTranspose2d
    else:
        raise ValueError(f"Unsupported num of image dimension '{len(inputs)}'")
