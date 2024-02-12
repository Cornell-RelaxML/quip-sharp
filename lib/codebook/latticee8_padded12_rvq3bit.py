"""
E8 3 bit.
Made from 2 bit E8P + 1 bit E8 with RVQ.
"""
import itertools
import math
from functools import cache

import quiptools_cuda
import torch
from torch import nn

from lib.utils.matmul_had import matmul_hadU_cuda, matmul_hadUt_cuda

_E8P_CODESZ = 8


def get_norm12():
    # 29 elements of norm 12 in E8 + 1/4
    return torch.tensor([
        [3, 1, 1, 1, 3, 3, 3, 3],
        [1, 3, 1, 1, 3, 3, 3, 3],
        [1, 1, 3, 1, 3, 3, 3, 3],
        [1, 1, 1, 3, 3, 3, 3, 3],
        [3, 3, 3, 1, 3, 3, 1, 1],
        [3, 3, 3, 1, 3, 1, 3, 1],
        [3, 3, 3, 1, 1, 3, 3, 1],
        [3, 3, 3, 1, 3, 1, 1, 3],
        [3, 3, 3, 1, 1, 3, 1, 3],
        [3, 3, 3, 1, 1, 1, 3, 3],
        [3, 3, 1, 3, 3, 3, 1, 1],
        [3, 3, 1, 3, 3, 1, 3, 1],
        [3, 3, 1, 3, 1, 3, 3, 1],
        [3, 3, 1, 3, 3, 1, 1, 3],
        [3, 3, 1, 3, 1, 3, 1, 3],
        [3, 3, 1, 3, 1, 1, 3, 3],
        [3, 1, 3, 3, 3, 3, 1, 1],
        [3, 1, 3, 3, 3, 1, 3, 1],
        [3, 1, 3, 3, 1, 3, 3, 1],
        [3, 1, 3, 3, 3, 1, 1, 3],
        [3, 1, 3, 3, 1, 3, 1, 3],
        [1, 3, 3, 3, 1, 1, 3, 3],
        [1, 3, 3, 3, 3, 3, 1, 1],
        [1, 3, 3, 3, 3, 1, 3, 1],
        [1, 3, 3, 3, 1, 3, 3, 1],
        [1, 3, 3, 3, 3, 1, 1, 3],
        [1, 3, 3, 3, 1, 3, 1, 3],
        [1, 1, 3, 3, 1, 3, 3, 3],
        [3, 3, 1, 1, 3, 3, 3, 1],
    ]) / 2


def get_packed_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * 8).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1)**2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    norm12 = get_norm12()
    cba = torch.concat([d8abs, norm12], dim=0)
    cba = cba[:, [0, 2, 4, 6, 1, 3, 5, 7]]
    cba[:, 7] *= (1 - 2 * (cba.sum(1) % 2))
    cba = cba * 2 + 8
    cba = cba.to(torch.int32)
    acc = cba[:, 0]
    for i in range(7):
        acc = acc | (cba[:, (i + 1)] << ((i + 1) * 4))
    return acc


def get_abs_grid():
    intr = torch.arange(-4, 4)
    d8 = torch.cartesian_prod(*[intr] * _E8P_CODESZ).float() + 1 / 2
    d8m2 = (d8.sum(dim=-1) % 2 == 0)
    d8n = d8.norm(dim=-1)**2 <= 10
    d8abs = torch.unique(d8[sorted(torch.where(d8m2 * d8n)[0])].abs(), dim=0)
    norm12 = get_norm12()
    cba = torch.concat([d8abs, norm12], dim=0)
    return cba


def get_full_grid(packed_abs_grid):
    synth_codebook = torch.zeros(1 << 16, 8)
    parity_idx = []
    shuffle_map = [0, 4, 1, 5, 2, 6, 3, 7]
    for c in range(1 << 16):
        signs = c & 255
        abs = c >> 8
        parity = 0
        for i in range(8):
            parity = parity ^ ((signs >> i) & 1)
        signs = signs ^ parity
        abs_code = packed_abs_grid[abs].item()
        for i in range(8):
            ii = shuffle_map[i]
            synth_codebook[c, i] = (((abs_code >> (4 * ii)) & 15) - 8) * 0.5
            if ((signs >> ii) & 1):
                synth_codebook[c, i] *= -1
        if parity:
            synth_codebook[c, :] -= 0.25
            parity_idx.append(c)
        else:
            synth_codebook[c, :] += 0.25
    return synth_codebook, torch.arange(1 << 16), parity_idx


_E8P_PACKED_ABS_CACHED = get_packed_abs_grid()
_E8P_GRID, _E8P_GRID_IDX, _PARITY_IDX = get_full_grid(_E8P_PACKED_ABS_CACHED)


def get_e81bgrid():
    intr = torch.arange(-4, 4)
    hintr = intr + 1 / 2

    gintr = torch.cartesian_prod(*[intr] * 8)
    ghintr = torch.cartesian_prod(*[hintr] * 8)

    ge8 = torch.concat([gintr, ghintr], dim=0)
    ge8m2 = (ge8.sum(dim=-1) % 2 == 0)
    ge8n = ge8.norm(dim=-1)**2 <= 2

    e8 = ge8[torch.where(ge8m2 * ge8n)[0]]

    norm4 = torch.tensor([
        [2, 0, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0, 0],
        [0, 0, 0, 2, 0, 0, 0, 0],
        [0, 0, 0, 0, 2, 0, 0, 0],
        [0, 0, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 0, 0, 2],
        [-2, 0, 0, 0, 0, 0, 0, 0],
        [0, -2, 0, 0, 0, 0, 0, 0],
        [0, 0, -2, 0, 0, 0, 0, 0],
        [0, 0, 0, -2, 0, 0, 0, 0],
        [0, 0, 0, 0, -2, 0, 0, 0],
        [0, 0, 0, 0, 0, -2, 0, 0],
        [0, 0, 0, 0, 0, 0, -2, 0],
        #[0, 0, 0, 0, 0, 0, 0, -2],
    ])

    e8 = torch.concat([e8, norm4], dim=0)

    return e8


_E81B_CACHED = get_e81bgrid()
_E81B_NORM_CACHED = _E81B_CACHED.norm(dim=-1)**2


class E8P12RVQ3B_codebook(nn.Module):

    def __init__(self, inference=False):
        super(E8P12RVQ3B_codebook, self).__init__()
        self.opt_scale = 0.98
        self.codesz = _E8P_CODESZ
        self.idx_dtype = torch.int64
        self.packsz = 8 / 3  # fudged, the second half of Qidxs is the residual
        self.pack_out = False
        self.version = 0
        self.opt_resid_scale = 2.04

        self.register_buffer('grid_packed_abs', _E8P_PACKED_ABS_CACHED)
        self.register_buffer('e81b_grid', _E81B_CACHED)

        if not inference:
            self.register_buffer('grid', _E8P_GRID)
            self.register_buffer('grid_norm', _E8P_GRID.norm(dim=-1)**2)
            grid_part = _E8P_GRID[_PARITY_IDX] + 0.25
            grid_part = grid_part[
                torch.where(
                    ((grid_part[:, :7] < 0).sum(dim=-1) <= 1) * \
                    (grid_part[:, :7].min(dim=-1).values >= -0.5)
                )[0]]
            self.register_buffer('grid_part', grid_part)
            self.register_buffer('grid_part_norm', grid_part.norm(dim=-1)**2)
            abs_grid = get_abs_grid()
            self.register_buffer('grid_abs_odd', abs_grid.sum(dim=-1) % 2 == 1)
            self.register_buffer(
                'part_abs_map',
                self.round(grid_part.abs(), abs_grid,
                           abs_grid.norm(dim=-1)**2)[1])
            self.register_buffer('bit_map', 2**torch.arange(8))
            self.register_buffer('e81b_grid_norm', _E81B_NORM_CACHED)
            '''
            self.to('cuda')
            samples = torch.distributions.multivariate_normal.MultivariateNormal(
                torch.zeros(8), torch.eye(8)).rsample([1000000]).cuda()

            from scipy.optimize import minimize_scalar
            def opt_err_cvx(fn):
                res = minimize_scalar(fn, bounds=(0.1, 100))
                scale = res.x.item()
                err = res.fun
                return err, scale
            
            for s in torch.arange(0.8, 1.1, 0.01):
                def test_sr(sr):
                    hatwr = self.quantize(samples*s, False, sr)/s
                    return (hatwr - samples).norm().cpu()**2
                err, scale = opt_err_cvx(test_sr)
                print(s, scale, err)
            exit()
            '''

    def round(self, X, grid, grid_norm):
        assert X.shape[-1] == self.codesz
        Xqidx = (2 * X @ grid.T - grid_norm).argmax(-1)
        return grid[Xqidx], Xqidx

    def fast_quantize_part(self, X, parity):
        X_part = torch.abs(X)
        X_odd = torch.where((X < 0).sum(dim=-1) % 2 != 0)[0]
        X_part[X_odd, 7] = -X_part[X_odd, 7]
        mask = 1 - 2 * (X < 0).to(torch.float32)
        mask[X_odd, 7] = -mask[X_odd, 7]
        roundout, Xqidx = self.round(X_part, self.grid_part,
                                     self.grid_part_norm)
        vals = roundout * mask
        err = (X - vals).norm(dim=-1)
        abs_idx = self.part_abs_map[Xqidx]
        sign_mask = (((roundout < 0) ^ (mask < 0))[:,
                                                   [0, 2, 4, 6, 1, 3, 5, 7]])
        sign_mask[:, 7] = sign_mask[:, 7] ^ self.grid_abs_odd[abs_idx]
        sign_mask[:, 0] = sign_mask[:, 0] ^ parity
        mask_idx = (sign_mask * self.bit_map).sum(dim=-1).int()
        idx = (abs_idx << 8) + mask_idx
        return vals, idx, err

    def quantize_e8p(self, X):
        X_plus = X + 1 / 4  # quantize X to D8^ - 1/4
        X_minus = X - 1 / 4  # quantize X to D8^ + 1/4

        plus_vals, plus_idx, plus_err = self.fast_quantize_part(X_plus, True)
        minus_vals, minus_idx, minus_err = self.fast_quantize_part(
            X_minus, False)

        which = plus_err < minus_err
        final_vals = torch.where(which.unsqueeze(-1), plus_vals - 1 / 4,
                                 minus_vals + 1 / 4)
        final_idx = torch.where(which, plus_idx, minus_idx)
        return final_vals, final_idx

    def quantize(self, X, return_idx=True, resid_scale_override=-1, **kwargs):
        init_vals, init_idxs = self.quantize_e8p(X)
        resid_scale = resid_scale_override if resid_scale_override > 0 else self.opt_resid_scale

        resid = (X - init_vals) * resid_scale
        resid_vals, resid_idxs = self.round(resid, self.e81b_grid,
                                            self.e81b_grid_norm)
        final_vals = init_vals + resid_vals / resid_scale
        final_idxs = (init_idxs << 16) + resid_idxs
        if return_idx:
            return final_vals, final_idxs
        return final_vals

    def maybe_pack_idxs(self, idxs):
        init_idxs = idxs >> 16
        resid_idxs = idxs & ((1 << 16) - 1)

        def pack_e8p(idxs):
            m, n = idxs.shape
            idxs = idxs.view(m // 2, 2, (n * 8) // 16,
                             2).transpose(1, 2).contiguous()

            abs32 = (idxs[:, :, 0, 0] >> 8) + \
                ((idxs[:, :, 1, 0] >> 8) << 8) + \
                ((idxs[:, :, 0, 1] >> 8) << 16) + \
                ((idxs[:, :, 1, 1] >> 8) << 24)

            sign32 = torch.zeros(abs32.shape,
                                 dtype=abs32.dtype,
                                 device=abs32.device)
            for i in range(4):
                wt = idxs[:, :, i % 2, i // 2]
                for j in range(8):
                    sign32 += ((wt >> j) & 1) << (4 * j + i)

            output = (sign32 << 32) + abs32
            output = output.reshape(m // 16, 8, n // 8,
                                    4).transpose(1, 2).contiguous()
            return output.view(m, n // 4)

        def pack_e81b(idxs):
            accum = idxs[:, 0::8]
            for i in range(1, 8):
                accum += idxs[:, i::8] << (8 * i)
            return accum

        return torch.concat(
            [pack_e8p(init_idxs), pack_e81b(resid_idxs)], dim=-1)

    def by_idxs(self, idxs, **kwargs):
        split = idxs.shape[-1] * 2 // 3
        init_idxs = idxs[:, :split].contiguous()
        resid_idxs = idxs[:, split:].contiguous()
        m, n = init_idxs.shape
        W_decompressed = quiptools_cuda.decompress_packed_e8p(
            init_idxs.view(m // 16, n // 2, 8, 4), self.grid_packed_abs)

        W_resid_decompressed = torch.zeros(
            resid_idxs.shape[0],
            64 * resid_idxs.shape[-1],
            device=resid_idxs.device,
            dtype=torch.float16,
        )

        quiptools_cuda.decompress_e81b_packed(resid_idxs,
                                              self.e81b_grid.to(torch.float16),
                                              W_resid_decompressed)
        return W_decompressed + W_resid_decompressed / self.opt_resid_scale


class QuantizedE8P12RVQ3BLinear(nn.Module):

    def __init__(self, device):
        super().__init__()
        self.codebook = E8P12RVQ3B_codebook(inference=True).to(
            torch.float16).to(device)
        self.scale = 32

    def maybe_unpack_idxs(self, idxs):
        split = idxs.shape[-1] * 2 // 3
        return (idxs[:, :split].contiguous(), idxs[:, split:].contiguous())

    def cache_WH(self,
                 n,
                 m,
                 Qidxs_list,
                 had_left,
                 had_right,
                 K_left,
                 K_right,
                 resid_scale_override=-1,
                 **kwargs):
        W_decompressed = quiptools_cuda.decompress_packed_e8p(
            Qidxs_list[0].view(m // 16, n // 64, 8, 4),
            self.codebook.grid_packed_abs)

        W_resid_decompressed = torch.zeros(Qidxs_list[1].shape[0],
                                           64 * Qidxs_list[1].shape[-1],
                                           device=Qidxs_list[1].device,
                                           dtype=torch.float16)

        quiptools_cuda.decompress_e81b_packed(Qidxs_list[1],
                                              self.codebook.e81b_grid,
                                              W_resid_decompressed)
        resid_scale = resid_scale_override if resid_scale_override > 0 else \
            self.codebook.opt_resid_scale

        self.W = matmul_hadU_cuda(
            matmul_hadU_cuda(
                (W_decompressed + W_resid_decompressed / resid_scale).float() /
                self.scale,
                had_left,
                K_left,
            ).T,
            had_right,
            K_right,
        ).to(torch.float16)

    def forward(self,
                input,
                Qidxs_list,
                SU,
                SV,
                had_left,
                had_right,
                K_left,
                K_right,
                rank=-1,
                A=None,
                B=None,
                rescale_WH=False,
                scaleWH=None,
                resid_scale_override=-1,
                train_mode=False,
                **kwargs):

        n, m = len(SU), len(SV)

        x = input.view(-1, n).to(torch.float32)
        if rescale_WH:
            x /= scaleWH
        x = x * SU

        if train_mode:
            x = (x.to(torch.float16) @ self.W).float()
        else:
            x = matmul_hadUt_cuda(x, had_left, K_left) / self.scale

            if rank > 0:
                Bx = x @ B.t().to(torch.float32)
                ABx = Bx @ A.t().to(torch.float32)

            resid_scale = resid_scale_override if resid_scale_override > 0 else \
                self.codebook.opt_resid_scale

            x16 = x.to(torch.float16)
            if x.shape[0] == 1:
                x_padded = torch.zeros(8,
                                       x16.shape[1],
                                       dtype=torch.float16,
                                       device=x16.device)
                x_padded[0] = x16[0]
                z = torch.zeros(8,
                                m,
                                dtype=torch.float32,
                                device=x_padded.device)
                quiptools_cuda.lookupmatmul_e81b_k8(x_padded / resid_scale,
                                                    Qidxs_list[1],
                                                    self.codebook.e81b_grid, z)

                x = quiptools_cuda.decode_matvec_e8p(
                    x16[0], Qidxs_list[0].view(m // 16, n // 64, 8, 4),
                    self.codebook.grid_packed_abs).to(torch.float32) + z[0]

            else:
                W_decompressed = quiptools_cuda.decompress_packed_e8p(
                    Qidxs_list[0].view(m // 16, n // 64, 8, 4),
                    self.codebook.grid_packed_abs)

                W_resid_decompressed = torch.zeros(Qidxs_list[1].shape[0],
                                                   64 *
                                                   Qidxs_list[1].shape[-1],
                                                   device=Qidxs_list[1].device,
                                                   dtype=torch.float16)

                quiptools_cuda.decompress_e81b_packed(Qidxs_list[1],
                                                      self.codebook.e81b_grid,
                                                      W_resid_decompressed)

                x = (x16 @ (W_decompressed +
                            W_resid_decompressed / resid_scale).T).to(
                                torch.float32)

            if rank > 0:
                x = x + ABx.to(torch.float32)

            x = matmul_hadU_cuda(x, had_right, K_right)

        x = x * SV * self.scale

        output = x.view(*input.shape[:-1], m)
        return output
