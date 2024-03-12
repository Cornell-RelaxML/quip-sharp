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
