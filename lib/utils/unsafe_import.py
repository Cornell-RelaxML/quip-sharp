# functions in this file cause circular imports so they cannot be loaded into __init__

from model.graph_wrapper import get_graph_wrapper
from model.llama import LlamaForCausalLM
import transformers

def model_from_hf_path(path, use_cuda_graph=True, use_flash_attn=True):
    def maybe_wrap(use_cuda_graph):
        return (lambda x: get_graph_wrapper(x)) if use_cuda_graph else (lambda x: x)
    
    if 'meta' in path:
        model = maybe_wrap(use_cuda_graph)(transformers.LlamaForCausalLM).from_pretrained(
            path, torch_dtype='auto', low_cpu_mem_usage=True, device_map='auto').half()
        model_str = path
    else:
        model = maybe_wrap(use_cuda_graph)(LlamaForCausalLM).from_pretrained(
            path, torch_dtype='auto', low_cpu_mem_usage=True, use_flash_attention_2=use_flash_attn, device_map='auto').half()
        model_str = transformers.LlamaConfig.from_pretrained(path)._name_or_path

    return model, model_str
