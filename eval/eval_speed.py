import argparse
import os
import time

import glog
import torch
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoTokenizer

from lib.utils.unsafe_import import model_from_hf_path

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', default='meta-llama/Llama-2-70b-hf', type=str)
parser.add_argument('--batch_size', default=1, type=int)
parser.add_argument('--seqlen', default=1, type=int)
parser.add_argument('--samples', default=100, type=int)
parser.add_argument('--no_use_cuda_graph', action='store_true')
parser.add_argument('--no_use_flash_attn', action='store_true')


def main(args):
    model, model_str = model_from_hf_path(
        args.hf_path,
        use_cuda_graph=not args.no_use_cuda_graph,
        use_flash_attn=not args.no_use_flash_attn)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    prompt = 'It is a truth universally acknowledged that'
    inputs = tokenizer(prompt, return_tensors='pt')
    token = inputs['input_ids'][0:1,
                                0:1].cuda().repeat(args.batch_size,
                                                   args.seqlen)
    model(token, use_cache=False)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(args.samples):
        model(token, use_cache=False)
    torch.cuda.synchronize()
    end = time.time()
    print('TIME', (end - start) / args.samples)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)
