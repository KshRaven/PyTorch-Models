
from Models.sub import BufferEmbedding, BufferEncoding, DownStream, UpStream, TransformerBase, Transpose
from Models.util.storage import save, load

import torch
import torch.nn as nn

from torch import Tensor, distributions


class AutoEncoder(nn.Module):
    def __init__(self, max_seq_len: int, inputs: int, embed_size: int, kernel_size: int,
                 stream_layers: int, trans_layers: int, fwd_layers: int,
                 heads: int, kv_heads: int = None, differential=False,
                 bias=True, device='cpu', dtype=torch.float32, **options):
        super(AutoEncoder, self).__init__()
        self.max_seq_len = max_seq_len
        self.inputs = inputs
        self.outputs = options.get('outputs', embed_size)
        self.embed_size = embed_size
        self.kernel_size = kernel_size
        self.stride = options.get('stride', 1)
        self.stream_layers = stream_layers
        self.trans_layers = trans_layers
        self.norm_groups = options.get('norm_groups', 1)
        self.affine = True
        self.epsilon = 1e-9
        self.constant = 10000.0
        self.probabilistic = options.get('probabilistic', False)

        self.encoder_embed = nn.Sequential(
            BufferEmbedding(inputs, embed_size, True, 'continuous', device, dtype),
            Transpose(),
            DownStream(embed_size, kernel_size, self.norm_groups, stream_layers, bias, device, dtype,
                       stride=self.stride),
            Transpose(),
        )
        self.encoder_tx = TransformerBase(
            max_seq_len, embed_size, trans_layers, heads, kv_heads, differential, True, bias, device, dtype,
            normalize=True, residual=True, auto_single=False, affine=self.affine, fwd_exp=4,
            constant=self.constant, epsilon=self.epsilon,
        )
        self.encoder_fwd = nn.Sequential(
            nn.LayerNorm((embed_size,), self.epsilon, self.affine, False, device, dtype),
            nn.Linear(embed_size, self.outputs, True, device, dtype),
        )

        self.decoder_embed = nn.Sequential(
            BufferEmbedding(self.outputs, embed_size, True, 'continuous', device, dtype),
            Transpose(),
            UpStream(embed_size, kernel_size, self.norm_groups, stream_layers, bias, device, dtype,
                     stride=self.stride),
            Transpose(),
        )
        self.decoder_tx = TransformerBase(
            max_seq_len, embed_size, trans_layers, heads, kv_heads, differential, True, bias, device, dtype,
            normalize=True, residual=True, auto_single=False, affine=self.affine, fwd_exp=4,
            constant=self.constant, epsilon=self.epsilon,
        )
        self.decoder_fwd = nn.Sequential(
            nn.LayerNorm((embed_size,), self.epsilon, self.affine, False, device, dtype),
            nn.Linear(embed_size, inputs * 2, True, device, dtype),
        )
        self.decoder_pass = nn.Sequential(
            nn.LayerNorm((embed_size,), self.epsilon, self.affine, False, device, dtype),
            nn.Linear(embed_size, self.outputs * 2, True, device, dtype),
        )
        self.eval()

    def encode(self, tensor: Tensor, verbose: int = False, single: bool = False, get: bool = False) -> Tensor:
        squeeze = tensor.ndim == 2
        if squeeze:
            tensor = tensor.unsqueeze(0)
        tensor = self.encoder_embed(tensor)
        tensor = self.encoder_tx(tensor, verbose=verbose, single=single, get=get)
        # tensor = torch.cat(torch.std_mean(tensor, dim=-2), dim=-1)
        tensor = self.encoder_fwd(tensor)
        if single:
            tensor = tensor.squeeze(-2)
        if squeeze:
            tensor = tensor.squeeze(0)
        return tensor

    def decode(self, tensor: Tensor, verbose: bool = False, single: bool = False, get: bool = False, type=0) -> Tensor:
        assert tensor.ndim >= 2
        squeeze = tensor.ndim == 2
        if squeeze:
            tensor = tensor.unsqueeze(0)
        # tensor = self.decoder_rev(tensor)
        # mean, log_std = torch.chunk(tensor.unsqueeze(-2).expand(-1, self.max_seq_len, self.squeeze), chunks=2, dim=-1)
        # tensor = mean + torch.randn_like(log_std) * torch.exp(log_std)
        tensor = self.decoder_embed(tensor)
        tensor = self.decoder_tx(tensor, verbose=verbose, single=single, get=get)

        def call(t: Tensor, method: callable):
            t = method(t)
            mean, log_std = t.chunk(2, dim=-1)
            if self.probabilistic:
                std = torch.exp(log_std)
                if self.training:
                    t = mean + std * torch.randn_like(std)
                else:
                    t = distributions.Normal(mean, std).sample()
            else:
                t = mean
            if single:
                t = t.squeeze(-2)
            if squeeze:
                t = t.squeeze(0)
            return t

        if type == 0:
            return call(tensor, self.decoder_fwd)
        elif type == 1:
            return call(tensor, self.decoder_pass)
        elif type == 2:
            return call(tensor, self.decoder_fwd), call(tensor, self.decoder_pass)
        else:
            raise NotImplementedError()

    def learn(self, inputs, type=0):
        return self.decode(self.encode(inputs), type=type)

    def forward(self, tensor: Tensor, verbose: int = False, single: bool = True, get: bool = False):
        return self.encode(tensor, verbose, single, get)

    def save(self, filename: str, file_no: int = None, directory: str = None, sub_directory: str = None,
             verbose: int = None):
        data = self.state_dict()
        save(
            data, filename, directory=directory, subdirectory=sub_directory, file_no=file_no, debug=verbose,
            items_name=f"{self.__class__.__name__} Model"
        )

    def load(self, filename: str, file_no: int = None, directory: str = None, sub_directory: str = None,
             verbose: int = None):
        data = load(
            filename, directory=directory, subdirectory=sub_directory, file_no=file_no, debug=verbose,
            items_name=f"{self.__class__.__name__} Model"
        )
        if data is not None:
            self.load_state_dict(data)
