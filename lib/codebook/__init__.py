import torch
import quiptools_cuda

torch.library.define("quip_lib::decode_matvec_e8p", "(Tensor x, Tensor Qidxs, Tensor grid_packed_abs, int m, int n) -> Tensor")

@torch.library.impl_abstract("quip_lib::decode_matvec_e8p")
def decode_matvec_e8p_abstract(
        x: torch.Tensor,
        Qidxs: torch.Tensor,
        grid_packed_abs: torch.Tensor,
        m: int, n: int) -> torch.Tensor:
    return x.new_empty(m, dtype=torch.float32, device=x.device)

@torch.library.impl("quip_lib::decode_matvec_e8p", "cuda")
def decode_matvec_e8p_cuda(
        x: torch.Tensor,
        Qidxs: torch.Tensor,
        grid_packed_abs: torch.Tensor,
        m: int, n: int) -> torch.Tensor:
    return quiptools_cuda.decode_matvec_e8p(x, Qidxs, grid_packed_abs)


torch.library.define("quip_lib::decompress_packed_e8p", "(Tensor Qidxs, Tensor grid_packed_abs, int m, int n) -> Tensor")

@torch.library.impl_abstract("quip_lib::decompress_packed_e8p")
def decompress_packed_e8p_abstract(
        Qidxs: torch.Tensor,
        grid_packed_abs: torch.Tensor,
        m: int, n: int) -> torch.Tensor:
    return Qidxs.new_empty(m, n, dtype=torch.float16, device=Qidxs.device)

@torch.library.impl("quip_lib::decompress_packed_e8p", "cuda")
def decompress_packed_e8p_cuda(
        Qidxs: torch.Tensor,
        grid_packed_abs: torch.Tensor,
        m: int, n: int) -> torch.Tensor:
    return quiptools_cuda.decompress_packed_e8p(Qidxs, grid_packed_abs)


from . import (latticee8_padded12, latticee8_padded12_rvq3bit,
               latticee8_padded12_rvq4bit)

# name: (id, codebook class)
codebook_id = {
    'E8P12': (7, latticee8_padded12.E8P12_codebook),
    'E8P12RVQ4B': (17, latticee8_padded12_rvq4bit.E8P12RVQ4B_codebook),
    'E8P12RVQ3B': (18, latticee8_padded12_rvq3bit.E8P12RVQ3B_codebook),
}

# id from above: quantized linear implementation
quantized_class = {
    7: latticee8_padded12.QuantizedE8P12Linear,
    17: latticee8_padded12_rvq4bit.QuantizedE8P12RVQ4BLinear,
    18: latticee8_padded12_rvq3bit.QuantizedE8P12RVQ3BLinear,
}

cache_permute_set = {}


def get_codebook(name):
    return codebook_id[name][1]()


def get_id(name):
    return codebook_id[name][0]


def get_quantized_class(id):
    return quantized_class[id]
