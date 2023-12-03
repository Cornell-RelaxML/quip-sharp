import torch
import torch.nn as nn
import quiptools_cuda
from lib.utils import dtype_from_str, get_hadK
from lib import codebook
import time

class QuantizedLinear(nn.Module):

    def __init__(self, in_features, out_features, codesz, packsz, idx_dtype, outlier_channel_split=False, rank=-1, rescale_WH=False):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.outlier_channel_split = outlier_channel_split
        self.rank = rank
        self.rescale_WH = rescale_WH

        if self.outlier_channel_split:
            self.register_buffer('ocs_dupe_inds', torch.arange(in_features))

        if self.rank > 0:
            self.register_buffer('A', torch.zeros(out_features, rank))
            self.register_buffer('B', torch.zeros(rank, in_features))
        else:
            self.A = None
            self.B = None

        if self.rescale_WH:
            self.register_buffer("scaleWH", torch.ones(in_features))
        else:
            self.scaleWH = None
            
        self.register_buffer("Qidxs", torch.zeros(
            out_features, in_features // (codesz*packsz), dtype=dtype_from_str(idx_dtype)))
        self.register_buffer("codebook_id", torch.tensor(0))
        self.register_buffer("SU", torch.ones(in_features))
        self.register_buffer("SV", torch.ones(out_features))
        self.register_buffer("Wscale", torch.ones(()))

        self.built_codebook_class = False
        self.built_graph = False

        had_left, K_left = get_hadK(in_features)
        had_right, K_right = get_hadK(out_features)
        self.register_buffer('had_left', had_left, persistent=False)
        self.register_buffer('had_right', had_right, persistent=False)
        self.K_left = K_left
        self.K_right = K_right
        self.packed = (packsz != 1)
        
    def forward(self, input):
        if not self.built_codebook_class:
            self.codebook_class = codebook.get_quantized_class(
                self.codebook_id.item())(self.Qidxs.device)
            self.built_codebook_class = True

        if self.outlier_channel_split:
            input = input[..., self.ocs_dupe_inds]

        return self.codebook_class(
            input,
            self.Qidxs, self.SU, self.SV, self.Wscale,
            self.had_left, self.had_right, self.K_left, self.K_right,
            rank=self.rank, A=self.A, B=self.B,
            rescale_WH=self.rescale_WH, scaleWH=self.scaleWH, packed=self.packed)
