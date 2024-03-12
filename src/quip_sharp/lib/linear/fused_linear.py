import torch
import torch.nn as nn


class FusedLinear(nn.Linear):

    def __init__(self, fuse_dim, fuse_sizes, *nnL_args, **nnL_kwargs):
        super(FusedLinear, self).__init__(*nnL_args, **nnL_kwargs)
        self.fuse_dim = fuse_dim
        self.fuse_sizes = fuse_sizes
        self.n = len(self.fuse_sizes)

    def forward(self, input):
        fused_output = super(FusedLinear, self).forward(input)
        return torch.split(fused_output, self.fuse_sizes, self.fuse_dim)
