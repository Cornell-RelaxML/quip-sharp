# functions in this file cause circular imports so they cannot be loaded into __init__

import json
import os

import transformers

from quip_sharp.model.llama import LlamaForCausalLM

from . import graph_wrapper


def model_from_hf_path(path,
                       use_cuda_graph=True,
                       use_flash_attn=True,
                       device_map='auto'):

    def maybe_wrap(use_cuda_graph):
        return (lambda x: graph_wrapper.get_graph_wrapper(x)
                ) if use_cuda_graph else (lambda x: x)

    # AutoConfig fails to read name_or_path correctly
    bad_config = transformers.AutoConfig.from_pretrained(path)
    is_quantized = hasattr(bad_config, 'quantization_config')
    model_type = bad_config.model_type
    if is_quantized:
        if model_type == 'llama':
            model_str = transformers.LlamaConfig.from_pretrained(
                path)._name_or_path
            model_cls = LlamaForCausalLM
        else:
            raise Exception
    else:
        model_cls = transformers.AutoModelForCausalLM

    model = maybe_wrap(use_cuda_graph)(model_cls).from_pretrained(
        path,
        torch_dtype='auto',
        low_cpu_mem_usage=True,
        use_flash_attention_2=use_flash_attn,
        device_map=device_map).half()

    return model, model_str
