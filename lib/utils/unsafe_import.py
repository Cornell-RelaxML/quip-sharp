# functions in this file cause circular imports so they cannot be loaded into __init__

from model.graph_wrapper import get_graph_wrapper
from model.llama import LlamaForCausalLM as llama_fuse
from model.mistral import MistralForCausalLM
import json
import os
import transformers

def model_from_hf_path(path, use_cuda_graph=True, use_flash_attn=True):
    def maybe_wrap(use_cuda_graph):
        return (lambda x: get_graph_wrapper(x)) if use_cuda_graph else (lambda x: x)

    # AutoConfig fails to read name_or_path correctly
    bad_config = transformers.AutoConfig.from_pretrained(path)
    is_quantized = hasattr(bad_config, 'quip_params')
    model_type = bad_config.model_type
    if is_quantized:
        if model_type == 'llama':
            model_str = transformers.LlamaConfig.from_pretrained(path)._name_or_path
            model_cls = llama_fuse
        elif model_type == 'mistral':
            model_str = transformers.MistralConfig.from_pretrained(path)._name_or_path
            model_cls = MistralForCausalLM
        else:
            raise Exception
    else:
        model_str = path
        if model_type == 'llama':
            model_cls = transformers.LlamaForCausalLM
        elif model_type == 'mistral':
            model_cls = transformers.MistralForCausalLM
        else:
            raise Exception

    model = maybe_wrap(use_cuda_graph)(model_cls).from_pretrained(
        path, torch_dtype='auto', low_cpu_mem_usage=True, use_flash_attention_2=use_flash_attn, device_map='auto').half()
            
    return model, model_str
