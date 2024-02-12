import time

import quiptools_cuda
import torch
import torch.nn as nn

from lib import codebook
from lib.utils import dtype_from_str, get_hadK

from .quantized_linear import QuantizedLinear


class FusedQuantizedLinear(QuantizedLinear):

    def __init__(self, fuse_dim, fuse_sizes, *QL_args, **QL_kwargs):
        super(FusedQuantizedLinear, self).__init__(*QL_args, **QL_kwargs)
        self.fuse_dim = fuse_dim
        self.fuse_sizes = fuse_sizes
        self.register_buffer('fuse_scales', torch.ones(len(self.fuse_sizes)))
        self.n = len(self.fuse_sizes)

    def forward(self, input):
        fused_output = super(FusedQuantizedLinear, self).forward(input)
        split_outputs = torch.split(fused_output, self.fuse_sizes,
                                    self.fuse_dim)
        return tuple(split_outputs[i] * self.fuse_scales[i]
                     for i in range(self.n))
