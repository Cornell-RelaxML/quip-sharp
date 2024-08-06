import time

import quiptools_cuda
import torch
import torch.nn as nn

from lib import codebook
from lib.utils import clean, dtype_from_str, get_hadK


class QuantizedLinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
        codesz,
        packsz,
        pack_out,
        idx_dtype,
        codebook_version,
        rank=-1,
        rescale_WH=False,
        bias=False,
        resid_scale_override=-1,
        train_mode=False,
        grad_ckpt=False,
    ):
        super().__init__()
        assert rank == 0 # 7/22/2024 removed support for low rank correction
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.rescale_WH = rescale_WH
        self.resid_scale_override = resid_scale_override

        self.has_bias = bias
        if self.has_bias:
            self.register_buffer('bias', torch.ones(out_features))

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

        # direction we pack in, the code dimension is always in the in dimension
        if pack_out:
            self.register_buffer(
                "Qidxs",
                torch.zeros(int(out_features / packsz),
                            int(in_features / codesz),
                            dtype=dtype_from_str(idx_dtype)))
        else:
            self.register_buffer(
                "Qidxs",
                torch.zeros(out_features,
                            int(in_features / (codesz * packsz)),
                            dtype=dtype_from_str(idx_dtype)))

        self.register_buffer("codebook_id", torch.tensor(0))
        self.register_buffer("SU", torch.ones(in_features,
                                              dtype=torch.float16))
        self.register_buffer("SV", torch.ones(out_features,
                                              dtype=torch.float16))
        self.register_buffer("Wscale", torch.ones(()))

        self.built_codebook_class = False
        self.built_graph = False
        self.codebook_version = codebook_version

        had_left_T, K_left = get_hadK(in_features)
        if had_left_T is not None:
            had_left_T = had_left_T.T.contiguous()
        self.register_buffer('had_left_T', had_left_T, persistent=False)
        
        had_right, K_right = get_hadK(out_features)
        self.register_buffer('had_right', had_right, persistent=False)
        
        self.K_left = K_left
        self.K_right = K_right
        self.packed = (packsz != 1)
        self.train_mode = train_mode
        self.grad_ckpt = grad_ckpt

    def forward(self, input):
        if self.grad_ckpt:
            return self.ckpt_forward(input)
        return self.no_ckpt_forward(input)

    def ckpt_forward(self, input):
        return torch.utils.checkpoint.checkpoint(self.no_ckpt_forward,
                                                 input,
                                                 use_reentrant=True)

    def no_ckpt_forward(self, input):
        if not self.built_codebook_class:
            self.codebook_class = codebook.get_quantized_class(
                self.codebook_id.item())(self.Qidxs.device)
            if self.codebook_class.codebook.version != self.codebook_version:
                raise Exception(
                    f"Saved weights version ({self.codebook_version}) does not match the "\
                    f"codebook version ({self.codebook_class.codebook.version}). "\
                    "Please download the latest weights from https://huggingface.co/relaxml")

            Qidxs_dev = self.Qidxs.device
            self.Qidxs = self.Qidxs.cpu()
            split_qidxs = self.codebook_class.maybe_unpack_idxs(self.Qidxs)
            self.Qidxs_list = []
            for i in range(len(split_qidxs)):
                self.register_buffer(f'Qidxs_{i}',
                                     split_qidxs[i].to(Qidxs_dev))
                exec(f'self.Qidxs_list.append(self.Qidxs_{i})')
            del self.Qidxs

            # fuse Wscale into SV, legacy code for when Wscale != 1
            # new models have Wscale pre-fused into SV
            self.SV *= self.Wscale

            # cache hadamard transformed manifested weights
            if self.train_mode:
                self.codebook_class.cache_WH(
                    len(self.SU),
                    len(self.SV),
                    self.Qidxs_list,
                    self.had_left_T.T.contiguous(),
                    self.had_right,
                    self.K_left,
                    self.K_right,
                    resid_scale_override=self.resid_scale_override,
                )
                del self.Qidxs_list, self.had_left_T, self.had_right, self.K_left, self.K_right
                self.Qidxs_list = None
                self.had_left_T = None
                self.had_right = None
                self.K_left = None
                self.K_right = None
                clean()

            self.built_codebook_class = True

        result = self.codebook_class(
            input,
            self.Qidxs_list,
            self.SU,
            self.SV,
            self.had_left_T,
            self.had_right,
            self.K_left,
            self.K_right,
            rank=self.rank,
            A=self.A,
            B=self.B,
            rescale_WH=self.rescale_WH,
            scaleWH=self.scaleWH,
            packed=self.packed,
            resid_scale_override=self.resid_scale_override,
            train_mode=self.train_mode).to(input.dtype)
        if self.has_bias:
            return result + self.bias
        return result
