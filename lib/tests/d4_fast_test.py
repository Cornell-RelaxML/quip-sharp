import torch
from torch import nn
from torch import optim
from transformers import LlamaTokenizer, LlamaForCausalLM
from datasets import load_dataset
import os

from torch.profiler import profile, record_function, ProfilerActivity

import time
import numpy
import scipy
from scipy.stats import special_ortho_group

from copy import deepcopy

import indexsum8_cpp

from tqdm import tqdm

torch.set_grad_enabled(False)

# import random
# torch.manual_seed(123456)
# random.seed(123456)
# numpy.random.seed(123456)

# codebook for D4
D4_CB = torch.tensor([[0.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0], [-1.0, -1.0, -1.0, -1.0], [-1.0, -1.0, -1.0, 0.0],
                      [-1.0, -1.0, -1.0, 1.0], [-1.0, -1.0, 0.0, -1.0], [-1.0, -1.0, 0.0, 0.0], [-1.0, -1.0, 0.0, 1.0],
                      [-1.0, -1.0, 1.0, -1.0], [-1.0, -1.0, 1.0, 0.0], [-1.0, -1.0, 1.0, 1.0], [-1.0, 0.0, -1.0, -1.0],
                      [-1.0, 0.0, -1.0, 0.0], [-1.0, 0.0, -1.0, 1.0], [-1.0, 0.0, 0.0, -1.0], [-1.0, 0.0, 0.0, 0.0],
                      [-1.0, 0.0, 0.0, 1.0], [-1.0, 0.0, 1.0, -1.0], [-1.0, 0.0, 1.0, 0.0], [-1.0, 0.0, 1.0, 1.0],
                      [-1.0, 1.0, -1.0, -1.0], [-1.0, 1.0, -1.0, 0.0], [-1.0, 1.0, -1.0, 1.0], [-1.0, 1.0, 0.0, -1.0],
                      [-1.0, 1.0, 0.0, 0.0], [-1.0, 1.0, 0.0, 1.0], [-1.0, 1.0, 1.0, -1.0], [-1.0, 1.0, 1.0, 0.0],
                      [-1.0, 1.0, 1.0, 1.0], [0.0, -2.0, 0.0, 0.0], [0.0, -1.0, -1.0, -1.0], [0.0, -1.0, -1.0, 0.0],
                      [0.0, -1.0, -1.0, 1.0], [0.0, -1.0, 0.0, -1.0], [0.0, -1.0, 0.0, 0.0], [0.0, -1.0, 0.0, 1.0],
                      [0.0, -1.0, 1.0, -1.0], [0.0, -1.0, 1.0, 0.0], [0.0, -1.0, 1.0, 1.0], [0.0, 0.0, -2.0, 0.0],
                      [0.0, 0.0, -1.0, -1.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, -1.0, 1.0], [0.0, 0.0, 0.0, -2.0],
                      [0.0, 0.0, 0.0, -1.0], [-1.5, -0.5, -1.5, -0.5], [-1.5, -0.5, -1.5, 0.5],
                      [-1.5, -0.5, -0.5, -1.5], [-1.5, -0.5, -0.5, -0.5], [-1.5, -0.5, -0.5, 0.5],
                      [-1.5, -0.5, -0.5, 1.5], [-1.5, -0.5, 0.5, -1.5], [-1.5, -0.5, 0.5, -0.5], [-1.5, -0.5, 0.5, 0.5],
                      [-1.5, -0.5, 0.5, 1.5], [-1.5, -0.5, 1.5, -0.5], [-1.5, -0.5, 1.5, 0.5], [-1.5, 0.5, -1.5, -0.5],
                      [-1.5, 0.5, -1.5, 0.5], [-1.5, 0.5, -0.5, -1.5], [-1.5, 0.5, -0.5, -0.5], [-1.5, 0.5, -0.5, 0.5],
                      [-1.5, 0.5, -0.5, 1.5], [-1.5, 0.5, 0.5, -1.5], [-1.5, 0.5, 0.5, -0.5], [-1.5, 0.5, 0.5, 0.5],
                      [-1.5, 0.5, 0.5, 1.5], [-1.5, 0.5, 1.5, -0.5], [-1.5, 0.5, 1.5, 0.5], [-1.5, 1.5, -0.5, -0.5],
                      [-1.5, 1.5, 0.5, -0.5], [-1.5, 1.5, 0.5, 0.5], [-0.5, -1.5, -1.5, -0.5], [-0.5, -1.5, -1.5, 0.5],
                      [-0.5, -1.5, -0.5, -1.5], [-0.5, -1.5, -0.5, -0.5], [-0.5, -1.5, -0.5, 0.5],
                      [-0.5, -1.5, -0.5, 1.5], [-0.5, -1.5, 0.5, -1.5], [-0.5, -1.5, 0.5, -0.5], [-0.5, -1.5, 0.5, 0.5],
                      [-0.5, -1.5, 0.5, 1.5], [-0.5, -1.5, 1.5, -0.5], [-0.5, -1.5, 1.5, 0.5], [-0.5, -0.5, -1.5, -1.5],
                      [-0.5, -0.5, -1.5, -0.5], [-0.5, -0.5, -1.5, 0.5], [-0.5, -0.5, -1.5, 1.5],
                      [-0.5, -0.5, -0.5, -1.5], [-0.5, -0.5, -0.5, -0.5], [-0.5, -0.5, -0.5, 0.5],
                      [-0.5, -0.5, -0.5, 1.5], [-0.5, -0.5, 0.5, -1.5], [-0.5, -0.5, 0.5, -0.5], [-0.5, -0.5, 0.5, 0.5],
                      [-0.5, -0.5, 0.5, 1.5], [-0.5, -0.5, 1.5, -1.5], [-0.5, -0.5, 1.5, -0.5], [-0.5, -0.5, 1.5, 0.5],
                      [-0.5, -0.5, 1.5, 1.5], [-0.5, 0.5, -1.5, -1.5], [-0.5, 0.5, -1.5, -0.5], [-0.5, 0.5, -1.5, 0.5],
                      [-0.5, 0.5, -1.5, 1.5], [-0.5, 0.5, -0.5, -1.5], [-0.5, 0.5, -0.5, -0.5], [-0.5, 0.5, -0.5, 0.5],
                      [-0.5, 0.5, -0.5, 1.5], [-0.5, 0.5, 0.5, -1.5], [-0.5, 0.5, 0.5, -0.5], [-0.5, 0.5, 0.5, 0.5],
                      [-0.5, 0.5, 0.5, 1.5], [-0.5, 0.5, 1.5, -1.5], [-0.5, 0.5, 1.5, -0.5], [-0.5, 0.5, 1.5, 0.5],
                      [-0.5, 0.5, 1.5, 1.5], [-0.5, 1.5, -1.5, -0.5], [-0.5, 1.5, -1.5, 0.5], [-0.5, 1.5, -0.5, -1.5],
                      [-0.5, 1.5, -0.5, -0.5], [-0.5, 1.5, -0.5, 0.5], [-0.5, 1.5, -0.5, 1.5], [-0.5, 1.5, 0.5, -1.5],
                      [-0.5, 1.5, 0.5, -0.5], [-0.5, 1.5, 0.5, 0.5], [-0.5, 1.5, 0.5, 1.5], [-0.5, 1.5, 1.5, -0.5],
                      [-0.5, 1.5, 1.5, 0.5], [0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0],
                      [1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 1.0, -1.0], [1.0, 1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0],
                      [1.0, 1.0, 0.0, -1.0], [1.0, 1.0, -1.0, 1.0], [1.0, 1.0, -1.0, 0.0], [1.0, 1.0, -1.0, -1.0],
                      [1.0, 0.0, 1.0, 1.0], [1.0, 0.0, 1.0, 0.0], [1.0, 0.0, 1.0, -1.0], [1.0, 0.0, 0.0, 1.0],
                      [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, -1.0], [1.0, 0.0, -1.0, 1.0], [1.0, 0.0, -1.0, 0.0],
                      [1.0, 0.0, -1.0, -1.0], [1.0, -1.0, 1.0, 1.0], [1.0, -1.0, 1.0, 0.0], [1.0, -1.0, 1.0, -1.0],
                      [1.0, -1.0, 0.0, 1.0], [1.0, -1.0, 0.0, 0.0], [1.0, -1.0, 0.0, -1.0], [1.0, -1.0, -1.0, 1.0],
                      [1.0, -1.0, -1.0, 0.0], [1.0, -1.0, -1.0, -1.0], [0.0, 2.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0],
                      [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0, -1.0], [0.0, 1.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0, -1.0], [0.0, 1.0, -1.0, 1.0], [0.0, 1.0, -1.0, 0.0], [0.0, 1.0, -1.0, -1.0],
                      [0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 1.0, -1.0],
                      [0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 1.0], [1.5, 0.5, 1.5, 0.5], [1.5, 0.5, 1.5, -0.5],
                      [1.5, 0.5, 0.5, 1.5], [1.5, 0.5, 0.5, 0.5], [1.5, 0.5, 0.5, -0.5], [1.5, 0.5, 0.5, -1.5],
                      [1.5, 0.5, -0.5, 1.5], [1.5, 0.5, -0.5, 0.5], [1.5, 0.5, -0.5, -0.5], [1.5, 0.5, -0.5, -1.5],
                      [1.5, 0.5, -1.5, 0.5], [1.5, 0.5, -1.5, -0.5], [1.5, -0.5, 1.5, 0.5], [1.5, -0.5, 1.5, -0.5],
                      [1.5, -0.5, 0.5, 1.5], [1.5, -0.5, 0.5, 0.5], [1.5, -0.5, 0.5, -0.5], [1.5, -0.5, 0.5, -1.5],
                      [1.5, -0.5, -0.5, 1.5], [1.5, -0.5, -0.5, 0.5], [1.5, -0.5, -0.5, -0.5], [1.5, -0.5, -0.5, -1.5],
                      [1.5, -0.5, -1.5, 0.5], [1.5, -0.5, -1.5, -0.5], [1.5, -1.5, 0.5, 0.5], [1.5, -1.5, -0.5, 0.5],
                      [1.5, -1.5, -0.5, -0.5], [0.5, 1.5, 1.5, 0.5], [0.5, 1.5, 1.5, -0.5], [0.5, 1.5, 0.5, 1.5],
                      [0.5, 1.5, 0.5, 0.5], [0.5, 1.5, 0.5, -0.5], [0.5, 1.5, 0.5, -1.5], [0.5, 1.5, -0.5, 1.5],
                      [0.5, 1.5, -0.5, 0.5], [0.5, 1.5, -0.5, -0.5], [0.5, 1.5, -0.5, -1.5], [0.5, 1.5, -1.5, 0.5],
                      [0.5, 1.5, -1.5, -0.5], [0.5, 0.5, 1.5, 1.5], [0.5, 0.5, 1.5, 0.5], [0.5, 0.5, 1.5, -0.5],
                      [0.5, 0.5, 1.5, -1.5], [0.5, 0.5, 0.5, 1.5], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, -0.5],
                      [0.5, 0.5, 0.5, -1.5], [0.5, 0.5, -0.5, 1.5], [0.5, 0.5, -0.5, 0.5], [0.5, 0.5, -0.5, -0.5],
                      [0.5, 0.5, -0.5, -1.5], [0.5, 0.5, -1.5, 1.5], [0.5, 0.5, -1.5, 0.5], [0.5, 0.5, -1.5, -0.5],
                      [0.5, 0.5, -1.5, -1.5], [0.5, -0.5, 1.5, 1.5], [0.5, -0.5, 1.5, 0.5], [0.5, -0.5, 1.5, -0.5],
                      [0.5, -0.5, 1.5, -1.5], [0.5, -0.5, 0.5, 1.5], [0.5, -0.5, 0.5, 0.5], [0.5, -0.5, 0.5, -0.5],
                      [0.5, -0.5, 0.5, -1.5], [0.5, -0.5, -0.5, 1.5], [0.5, -0.5, -0.5, 0.5], [0.5, -0.5, -0.5, -0.5],
                      [0.5, -0.5, -0.5, -1.5], [0.5, -0.5, -1.5, 1.5], [0.5, -0.5, -1.5, 0.5], [0.5, -0.5, -1.5, -0.5],
                      [0.5, -0.5, -1.5, -1.5], [0.5, -1.5, 1.5, 0.5], [0.5, -1.5, 1.5, -0.5], [0.5, -1.5, 0.5, 1.5],
                      [0.5, -1.5, 0.5, 0.5], [0.5, -1.5, 0.5, -0.5], [0.5, -1.5, 0.5, -1.5], [0.5, -1.5, -0.5, 1.5],
                      [0.5, -1.5, -0.5, 0.5], [0.5, -1.5, -0.5, -0.5], [0.5, -1.5, -0.5, -1.5], [0.5, -1.5, -1.5, 0.5],
                      [0.5, -1.5, -1.5, -0.5]]).cuda()

D4_CB_half = D4_CB[0:128, :]

batch_size = 4
valset_size = 128
ctx_size = 2048
H_regularization = 1e-2
c_reg = 1e-2
loquip_rank = 32


class QuIPA2Linear(nn.Module):

    def __init__(self, DSC, scaleW, Qidxs, Qscales, Cppi, ipp_in, ipp_out, A, B, U1, U2, V1, V2):
        super().__init__()
        self.register_buffer("scaleW_DSC", scaleW / DSC)
        self.register_buffer("Qidxs", Qidxs)
        self.register_buffer("Qscales", Qscales)
        self.register_buffer("Cpp", torch.argsort(Cppi))
        self.register_buffer("pp_in", torch.argsort(ipp_in))
        self.register_buffer("ipp_out", ipp_out)
        self.A = nn.Parameter(A.to(torch.float32))
        self.B = nn.Parameter(B.to(torch.float32))
        self.U1 = nn.Parameter(U1)
        self.U2 = nn.Parameter(U2)
        self.V1 = nn.Parameter(V1)
        self.V2 = nn.Parameter(V2)
        self.register_buffer("D4_CB_half", D4_CB_half)

    def forward(self, input):
        (p1, _) = self.U1.shape
        (p2, _) = self.U2.shape
        (q1, _) = self.V1.shape
        (q2, _) = self.V2.shape
        (m, n) = self.Qidxs.shape
        # if (p1 * p2 == q1 * q2):
        #     return input
        # else:
        #     return torch.zeros(*input.shape[0:-1],self.Qidxs.shape[0], dtype=input.dtype, device=input.device)
        # return input @ torch.zeros(2*n,m,dtype=input.dtype,device=input.device)
        # return torch.zeros(*input.shape[0:-1],q1*q2,dtype=input.dtype,device=input.device)
        assert (input.shape[-1] == p1 * p2)
        x = input.view(-1, p1 * p2)
        # x = x[:,self.pp_in]
        x.mul_(self.scaleW_DSC)
        if (x.shape[0] == 1):
            x = x.view(p1, p2)
            # x = self.U1 @ x
            # x = x @ self.U2.t()
            x = x.view(1, p1 * p2)
        else:
            x = x.view(-1, p1, p2)
            # x = torch.tensordot(x,self.U1,dims=([1],[1]))
            # x = torch.tensordot(x,self.U2,dims=([1],[1]))
            # x = torch.einsum("bij,pi,qj->bpq",x,self.U1,self.U2)
            x = x.view(-1, p1 * p2)
        # Bx = x @ self.B.t()
        # ABx = Bx @ self.A.t()
        # x = x[:,self.Cpp]
        x = x.view(-1, p1 * p2 // 4, 4)
        x.mul_(self.Qscales[:, None])
        z = torch.zeros(x.shape[0], m, dtype=x.dtype, device=x.device)
        M = torch.zeros(8, x.shape[1], 128, dtype=x.dtype, device=x.device)
        sidx = 0
        while (x.shape[0] - sidx >= 8):
            torch.matmul(x[sidx:(sidx + 8), :, :], self.D4_CB_half.t(), out=M)
            indexsum8_cpp.indexsum8x8(self.Qidxs, M, z[sidx:(sidx + 8), :])
            sidx += 8
        while (x.shape[0] - sidx >= 4):
            torch.matmul(x[sidx:(sidx + 4), :, :], self.D4_CB_half.t(), out=M[0:4, :, :])
            indexsum8_cpp.indexsum8x4(self.Qidxs, M[0:4, :, :], z[sidx:(sidx + 4), :])
            sidx += 4
        while (x.shape[0] - sidx >= 2):
            torch.matmul(x[sidx:(sidx + 2), :, :], self.D4_CB_half.t(), out=M[0:2, :, :])
            indexsum8_cpp.indexsum8x2(self.Qidxs, M[0:2, :, :], z[sidx:(sidx + 2), :])
            sidx += 2
        while (x.shape[0] - sidx >= 1):
            torch.matmul(x[sidx, :, :], self.D4_CB_half.t(), out=M[0, :, :])
            indexsum8_cpp.indexsum8(self.Qidxs, M[0, :, :], z[sidx, :])
            sidx += 1
        # M = torch.zeros(x.shape[1],128,dtype=x.dtype,device=x.device)
        # for k in range(x.shape[0]):
        #     M = x[k,:,:] @ self.D4_CB_half.t()
        #     indexsum8_cpp.indexsum8(self.Qidxs, M, z[k,:])
        x = z
        # x = x + ABx
        assert (x.shape[-1] == q1 * q2)
        if (x.shape[0] == 1):
            x = x.view(q1, q2)
            # x = self.V1.t() @ x
            # x = x @ self.V2
            x = x.view(1, q1 * q2)
        else:
            x = x.view(-1, q1, q2)
            # x = torch.einsum("bij,ip,jq->bpq",x,self.V1,self.V2)
            # x = torch.tensordot(x,self.V1,dims=([1],[0]))
            # x = torch.tensordot(x,self.V2,dims=([1],[0]))
            x = x.view(-1, q1 * q2)
        # x = x[:,self.ipp_out]
        x = x.view(*input.shape[0:-1], q1 * q2)
        return x


# class QuIPA2Linear(nn.Module):
#     def __init__(self, DSC, scaleW, Qidxs, Qscales, Cppi, ipp_in, ipp_out, A, B, U1, U2, V1, V2):
#         super().__init__()
#         self.register_buffer("scaleW_DSC", scaleW / DSC)
#         self.register_buffer("Qidxs", Qidxs)
#         self.register_buffer("Qscales", Qscales)
#         self.register_buffer("Cpp", torch.argsort(Cppi))
#         self.register_buffer("pp_in", torch.argsort(ipp_in))
#         self.register_buffer("ipp_out", ipp_out)
#         self.A = nn.Parameter(A.to(torch.float32))
#         self.B = nn.Parameter(B.to(torch.float32))
#         self.U1 = nn.Parameter(U1)
#         self.U2 = nn.Parameter(U2)
#         self.V1 = nn.Parameter(V1)
#         self.V2 = nn.Parameter(V2)
#         self.register_buffer("D4_CB_half", D4_CB_half)

#     def forward(self, input):
#         (p1,_) = self.U1.shape
#         (p2,_) = self.U2.shape
#         (q1,_) = self.V1.shape
#         (q2,_) = self.V2.shape
#         (m,n) = self.Qidxs.shape
#         # return input @ torch.zeros(4*n,m,dtype=input.dtype,device=input.device)
#         # return torch.zeros(*input.shape[0:-1],q1*q2,dtype=input.dtype,device=input.device)
#         assert(input.shape[-1] == p1 * p2)
#         x = input.view(-1,p1*p2)
#         x = x[:,self.pp_in]
#         x = x * self.scaleW_DSC
#         if (x.shape[0] == 1):
#             x = x.view(p1,p2)
#             x = self.U1 @ x
#             x = x @ self.U2.t()
#             x = x.view(1,p1*p2)
#         else:
#             x = x.view(-1,p1,p2)
#             x = torch.tensordot(x,self.U1,dims=([1],[1]))
#             x = torch.tensordot(x,self.U2,dims=([1],[1]))
#             x = x.view(-1,p1*p2)
#         Bx = x @ self.B.t()
#         ABx = Bx @ self.A.t()
#         x = x[:,self.Cpp]
#         x = x.view(-1,p1*p2//4,4)
#         x.mul_(self.Qscales[:,None])
#         z = torch.zeros(x.shape[0],m,dtype=x.dtype,device=x.device)
#         M = torch.zeros(x.shape[1],128,dtype=x.dtype,device=x.device)
#         for k in range(x.shape[0]):
#             torch.matmul(x[k,:,:], self.D4_CB_half.t(), out=M)
#             indexsum8_cpp.indexsum8(self.Qidxs, M, z[k,:])
#         x = z
#         x = x + ABx
#         assert(x.shape[-1] == q1 * q2)
#         if (x.shape[0] == 1):
#             x = x.view(q1,q2)
#             x = self.V1.t() @ x
#             x = x @ self.V2
#             x = x.view(1,q1*q2)
#         else:
#             x = x.view(-1,q1,q2)
#             x = torch.tensordot(x,self.V1,dims=([1],[0]))
#             x = torch.tensordot(x,self.V2,dims=([1],[0]))
#             x = x.view(-1,q1*q2)
#         x = x[:,self.ipp_out]
#         x = x.view(*input.shape[0:-1],q1*q2)
#         return x


def sample_devset(dataset, tokenizer, size=128):
    devset = []
    while (len(devset) < size):
        tokens = tokenizer(dataset[torch.randint(len(dataset), ()).item()]['text'],
                           return_tensors='pt',
                           truncation=True,
                           max_length=2048).input_ids
        if (tokens.shape[1] == 2048):
            devset.append(tokens)
    devset = torch.vstack(devset)
    return devset


def load_quip(save_name):
    print(f"loading cached compressed layer from path \"{save_name}\"")
    (DSC, scaleW, Qidxs, Qscales, Cppi, ipp_in, ipp_out, A, B, U1, U2, V1, V2) = torch.load(save_name)
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    U1 = U1.to(torch.float32)
    U2 = U2.to(torch.float32)
    V1 = V1.to(torch.float32)
    V2 = V2.to(torch.float32)
    Qidxs = Qidxs.to(torch.int64)
    (m, n) = Qidxs.shape
    hatWr = torch.zeros(m, 4 * n, device=DSC.device)
    for k in range(n):
        hatWr[:, (4 * k):(4 * (k + 1))] = D4_CB[Qidxs[:, k], :] * Qscales[k]
    hatWr = hatWr[:, Cppi]
    hatWr.add_(A @ B)
    U = torch.kron(U1, U2)
    V = torch.kron(V1, V2)
    hatW = V.t() @ (hatWr * scaleW) @ U
    hatW = hatW / DSC[None, :]
    hatW = hatW[:, ipp_in][ipp_out, :]
    return hatW


def main():
    print("loading model...")
    model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf",
                                             local_files_only=True,
                                             torch_dtype="auto",
                                             low_cpu_mem_usage=True)
    tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf", local_files_only=True)
    print("loaded model!")
    tokenizer.pad_token = tokenizer.eos_token

    model = model.cuda()
    # model = model.to_bettertransformer()

    # warm up
    for ii in range(5):
        output = model(torch.tensor(0).to(torch.int64).reshape(1, 1).cuda(),
                       use_cache=False,
                       output_attentions=False,
                       output_hidden_states=False)

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    print("running model forward...")
    start = time.time()
    for ii in range(10):
        output = model(torch.tensor(0).to(torch.int64).reshape(1, 1).cuda(),
                       use_cache=False,
                       output_attentions=False,
                       output_hidden_states=False)
    end = time.time()
    print(output)
    print(f"elapsed: {end - start}")
    print("done!")

    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total"))

    import random
    torch.manual_seed(123456)
    random.seed(123456)
    numpy.random.seed(123456)

    print("generating some text...")
    start = time.time()

    prompt = """Hermione Granger could not believe that she was"""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        min_new_tokens=128,  # penalty_alpha=0.6, top_k=16, 
        return_dict_in_generate=
        True  # max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.7, top_k=8, return_dict_in_generate=True
    )
    token = outputs.sequences[0, input_length:]
    output_str = tokenizer.decode(token)
    print(output_str)

    end = time.time()
    print(f"\nelapsed: {end - start}\n\n")

    # model = model.reverse_bettertransformer()
    model = model.cpu()

    for (transformer_layer_index, transformer_layer) in enumerate(model.model.layers):
        # check that there are four layers, as expected
        assert (len([m for m in transformer_layer.modules() if isinstance(m, torch.nn.Linear)]) == 7)

        print(f"decompressing block {transformer_layer_index}")

        for lmi in range(7):
            lm = [m for m in transformer_layer.modules() if isinstance(m, torch.nn.Linear)][lmi]
            lm.a2savefile = f"rp_compressed/{transformer_layer_index}_{lmi}.pt"

        transformer_layer.self_attn.q_proj = QuIPA2Linear(
            *torch.load(transformer_layer.self_attn.q_proj.a2savefile, map_location=torch.device('cpu'))).half()
        transformer_layer.self_attn.k_proj = QuIPA2Linear(
            *torch.load(transformer_layer.self_attn.k_proj.a2savefile, map_location=torch.device('cpu'))).half()
        transformer_layer.self_attn.v_proj = QuIPA2Linear(
            *torch.load(transformer_layer.self_attn.v_proj.a2savefile, map_location=torch.device('cpu'))).half()
        transformer_layer.self_attn.o_proj = QuIPA2Linear(
            *torch.load(transformer_layer.self_attn.o_proj.a2savefile, map_location=torch.device('cpu'))).half()
        transformer_layer.mlp.gate_proj = QuIPA2Linear(
            *torch.load(transformer_layer.mlp.gate_proj.a2savefile, map_location=torch.device('cpu'))).half()
        transformer_layer.mlp.down_proj = QuIPA2Linear(
            *torch.load(transformer_layer.mlp.down_proj.a2savefile, map_location=torch.device('cpu'))).half()
        transformer_layer.mlp.up_proj = QuIPA2Linear(
            *torch.load(transformer_layer.mlp.up_proj.a2savefile, map_location=torch.device('cpu'))).half()

    # for i in range(len(model.model.layers)):
    #     model.model.layers[i] = FakeIdBlock()

    # hopefully doesn't run out of memory
    print("moving model to gpu...")
    model = model.cuda()
    # model = model.to_bettertransformer()
    print("model on gpu!")

    # model = torch.compile(model)

    # warm up
    for ii in range(5):
        output = model(torch.tensor(0).to(torch.int64).reshape(1, 1).cuda(),
                       use_cache=False,
                       output_attentions=False,
                       output_hidden_states=False)

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    print("running model forward...")
    start = time.time()
    for ii in range(10):
        output = model(torch.tensor(0).to(torch.int64).reshape(1, 1).cuda(),
                       use_cache=False,
                       output_attentions=False,
                       output_hidden_states=False)
    end = time.time()
    print(output)
    print(f"elapsed: {end - start}")
    print("done!")

    # print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total"))

    import random
    torch.manual_seed(123456)
    random.seed(123456)
    numpy.random.seed(123456)

    print("generating some text...")
    start = time.time()

    prompt = """Hermione Granger could not believe that she was"""
    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    input_length = inputs.input_ids.shape[1]
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        min_new_tokens=512,  # penalty_alpha=0.6, top_k=16, 
        return_dict_in_generate=
        True  # max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.7, top_k=8, return_dict_in_generate=True
    )
    token = outputs.sequences[0, input_length:]
    output_str = tokenizer.decode(token)
    print(output_str)

    end = time.time()
    print(f"\nelapsed: {end - start}\n\n")


if __name__ == "__main__":
    main()
