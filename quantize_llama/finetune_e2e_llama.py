import argparse
import copy
import datetime
import gc
import math
import os
import time

from tqdm import tqdm

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import glog
import torch
import torch.multiprocessing as mp
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import codebook, utils
from lib.algo import finetune, quip
from lib.utils.unsafe_import import model_from_hf_path

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=64, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--base_model', type=str)
parser.add_argument('--hf_path', type=str)
parser.add_argument('--ckpt_path', type=str)
parser.add_argument('--ft_lr', default=1e-5, type=float)
parser.add_argument('--ft_susv_lr', default=1e-4, type=float)
parser.add_argument('--ft_bs', default=8, type=int)
parser.add_argument('--ft_update_freq', default=1, type=int)
parser.add_argument('--ft_epochs', default=1, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=3, type=int)
parser.add_argument('--ft_train_mode', action='store_true')
parser.add_argument('--ft_grad_ckpt', action='store_true')
parser.add_argument('--ft_nshards', default=-1, type=int)


def get_llama_save_fn(args):

    def save_fn(shard_model):
        ct = 0
        for i in range(len(shard_model.shards)):
            for j in range(len(shard_model.shards[i].layers)):
                shard = shard_model.shards[i].layers[j]
                utils.save_susv(shard.self_attn.qkv_proj,
                                f'{args.ckpt_path}/{ct}_qkv.pt')
                utils.save_susv(shard.self_attn.o_proj,
                                f'{args.ckpt_path}/{ct}_o.pt')
                utils.save_susv(shard.mlp.upgate_proj,
                                f'{args.ckpt_path}/{ct}_up.pt')
                utils.save_susv(shard.mlp.down_proj,
                                f'{args.ckpt_path}/{ct}_down.pt')
                torch.save(
                    {
                        'input_layernorm':
                        shard.input_layernorm.weight,
                        'post_attention_layernorm':
                        shard.post_attention_layernorm.weight,
                    }, f'{args.ckpt_path}/{ct}_layernorm.pt')
                glog.info(f'wrote layer {ct}')
                ct += 1
        torch.save(
            {
                'lm_head': shard_model.output_layer[1].weight,
                'norm': shard_model.output_layer[0].weight,
            }, f'{args.ckpt_path}/lmhead.pt')

    return save_fn


def llama_arg_fn(output, args, kwargs):
    return (output[0], *args[1:]), kwargs


def get_emb(args, kwargs):
    return args[0]


def main(args):
    torch.set_grad_enabled(False)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    devset = utils.sample_rp1t(tokenizer, args.devset_size, args.ctx_size,
                               args.sample_proc)

    orig_model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                      torch_dtype='auto',
                                                      device_map='auto',
                                                      low_cpu_mem_usage=True)
    orig_logits = utils.calculate_logits(orig_model, devset, args.batch_size)
    orig_logits = orig_logits[:, :-1].contiguous().softmax(dim=-1).float()

    del orig_model
    utils.clean()

    quant_model = model_from_hf_path(args.hf_path,
                                     use_cuda_graph=False,
                                     use_flash_attn=False,
                                     device_map=None)[0].cpu()
    emb = quant_model.model.embed_tokens(devset)
    position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :] + \
        torch.zeros(args.ft_bs, args.ctx_size, dtype=torch.int32)
    attention_mask = _prepare_4d_causal_attention_mask(
        None, (args.ft_bs, args.ctx_size), emb[:args.ft_bs], 0)

    # construct shards
    nshards = torch.cuda.device_count(
    ) if args.ft_nshards < 0 else args.ft_nshards
    nlayers = len(quant_model.model.layers)
    shards = [nn.ModuleList([]) for _ in range(nshards)]
    for i in range(nshards):
        for j in range(int(nlayers * i / nshards),
                       int(nlayers * (i + 1) / nshards)):
            shards[i].append(quant_model.model.layers[j])
        shards[i] = {'device': i, 'arg_fn': llama_arg_fn, 'shard': shards[i]}
    output_layer = {
        'layer': nn.Sequential(quant_model.model.norm, quant_model.lm_head),
        'fn': get_emb
    }

    shard_model = utils.ShardTransformer(shards, output_layer,
                                         args.ft_grad_ckpt, args.ft_train_mode)
    shard_model.manifest(emb[:args.ft_bs],
                         position_ids=position_ids,
                         attention_mask=attention_mask)
    utils.clean()

    torch.set_grad_enabled(True)
    finetune.finetune_susv_e2e(shard_model, orig_logits, emb, position_ids,
                               attention_mask, get_llama_save_fn(args), args)


if __name__ == '__main__':
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    main(args)
