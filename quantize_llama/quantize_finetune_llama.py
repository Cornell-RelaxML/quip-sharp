import argparse
import os
import time

import glog

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

import torch
import torch.multiprocessing as mp
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_attn_mask_utils import \
    _prepare_4d_causal_attention_mask

from lib import codebook, utils
from lib.algo import finetune, quip
from lib.linear import FusedLinear
from model.llama import LlamaDecoderLayer

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_cpu_threads', default=8, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--devset_size', default=384, type=int)
parser.add_argument('--ctx_size', default=4096, type=int)
parser.add_argument('--save_path', type=str)
parser.add_argument('--hessian_path', type=str)
parser.add_argument('--base_model', type=str)
parser.add_argument('--sigma_reg', default=1e-2, type=float)
parser.add_argument('--sigma_reg2', default=1e-2, type=float)
parser.add_argument('--incoh_mode',
                    default='had',
                    type=str,
                    choices=['had', 'kron'])
parser.add_argument('--lora_rank',
                    default=0,
                    type=int,
                    help='if <=0 then turned off')
parser.add_argument('--scale_override', default=-1, type=float)
parser.add_argument('--resid_scale_override', default=-1, type=float)
parser.add_argument('--codebook', type=str)
parser.add_argument('--quip_tune_iters', default=10, type=int)
parser.add_argument('--use_fp64', action='store_true')
parser.add_argument('--full_svd', action='store_true')
parser.add_argument('--no_use_buffered', action='store_true')
parser.add_argument('--rescale_WH', action='store_true')
parser.add_argument('--sample_proc', default=1, type=int)
parser.add_argument('--lowmem_ldlq', action='store_true')
parser.add_argument('--ft_lr', default=5e-5, type=float)
parser.add_argument('--ft_susv_lr', default=5e-4, type=float)
parser.add_argument('--ft_bs', default=4, type=int)
parser.add_argument('--ft_update_freq', default=2, type=int)
parser.add_argument('--ft_epochs', default=5, type=int)
parser.add_argument('--ft_valid_freq', default=1, type=int)
parser.add_argument('--ft_valid_size', default=128, type=float)
parser.add_argument('--ft_early_stop', default=3, type=int)
parser.add_argument('--ft_train_mode', action='store_true')
parser.add_argument('--ft_grad_ckpt', action='store_true')


def check_exist(idx, args):
    suffix = ['qkv', 'o', 'up', 'down', 'layernorm']
    for _ in suffix:
        test = f'{args.save_path}/{idx}_{_}.pt'
        if not os.path.exists(test):
            return False
    return True


def quantize_llama_layer(layer, idx, cb, args, device, pre_orig_emb, orig_emb,
                         model_config):
    if check_exist(idx, args):
        return

    mixed_layer = LlamaDecoderLayer(model_config, idx).cpu()
    with torch.no_grad():
        weights = [
            layer.self_attn.q_proj.weight, layer.self_attn.k_proj.weight,
            layer.self_attn.v_proj.weight
        ]

        fused_qkv_proj = FusedLinear(-1, [_.shape[0] for _ in weights],
                                     weights[0].shape[1],
                                     sum([_.shape[0] for _ in weights]),
                                     bias=False)
        cur = 0
        for w in weights:
            fused_qkv_proj.weight[cur:cur + w.shape[0]].copy_(w)
            cur += w.shape[0]

        mixed_layer.self_attn.qkv_proj = fused_qkv_proj

        mixed_layer.self_attn.o_proj = layer.self_attn.o_proj

        weights = [layer.mlp.up_proj.weight, layer.mlp.gate_proj.weight]
        fused_upgate_proj = FusedLinear(-1, [_.shape[0] for _ in weights],
                                        weights[0].shape[1],
                                        sum([_.shape[0] for _ in weights]),
                                        bias=False)
        cur = 0
        for w in weights:
            fused_upgate_proj.weight[cur:cur + w.shape[0]].copy_(w)
            cur += w.shape[0]
        mixed_layer.mlp.upgate_proj = fused_upgate_proj

        mixed_layer.mlp.down_proj = layer.mlp.down_proj

        mixed_layer.input_layernorm.weight.copy_(layer.input_layernorm.weight)
        mixed_layer.post_attention_layernorm.weight.copy_(
            layer.post_attention_layernorm.weight)

    finetune.quantize_finetune_decoder_layer(mixed_layer,
                                             [('self_attn.qkv_proj', 'qkv'),
                                              ('self_attn.o_proj', 'o'),
                                              ('mlp.upgate_proj', 'up'),
                                              ('mlp.down_proj', 'down')], idx,
                                             cb, args, device, pre_orig_emb,
                                             orig_emb)

    torch.save(
        {
            'input_layernorm':
            mixed_layer.input_layernorm.weight,
            'post_attention_layernorm':
            mixed_layer.post_attention_layernorm.weight,
        }, f'{args.save_path}/{idx}_layernorm.pt')
    del mixed_layer


def main(args):
    dtype_ = torch.float64 if args.use_fp64 else torch.float32

    cb = codebook.get_codebook(args.codebook)

    model = AutoModelForCausalLM.from_pretrained(args.base_model,
                                                 torch_dtype='auto',
                                                 low_cpu_mem_usage=True)

    # save configs
    all_config = {'quant_args': args, 'model_config': model.config}
    quip_params = {
        'lora_rank': args.lora_rank,
        'rescale_WH': args.rescale_WH,
        'codebook': args.codebook,
        'codebook_version': cb.version,
        'codesz': cb.codesz,
        'idx_dtype': str(cb.idx_dtype),
        'packsz': cb.packsz,
        'resid_scale_override': args.resid_scale_override,
    }
    all_config['model_config'].update({'quip_params': quip_params})
    torch.save(all_config, os.path.join(args.save_path, 'config.pt'))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    glog.info('loaded model')

    devset = utils.sample_rp1t(tokenizer, args.devset_size, args.ctx_size,
                               args.sample_proc)
    glog.info('loaded dataset and devset')

    nproc = torch.cuda.device_count()
    orig_emb_cache = [model.model.embed_tokens(devset)]
    for _ in range(nproc):
        orig_emb_cache.append(
            torch.zeros(orig_emb_cache[0].shape,
                        dtype=orig_emb_cache[0].dtype,
                        device=orig_emb_cache[0].device))

    position_ids = torch.arange(args.ctx_size, dtype=torch.int32)[None, :] + \
        torch.zeros(args.batch_size, args.ctx_size, dtype=torch.int32)
    attention_mask = _prepare_4d_causal_attention_mask(
        None, (args.batch_size, args.ctx_size),
        orig_emb_cache[0][:args.batch_size], 0)

    cur_device = 0
    proc_list = [None for _ in range(nproc)]
    for i in range(len(model.model.layers)):
        glog.info(f'layer {i} gpu {cur_device}')
        if proc_list[cur_device] is not None:
            proc_list[cur_device].join()
            if cur_device == 0:
                orig_emb_cache[0].copy_(orig_emb_cache[-1])
        if cur_device + 1 < nproc and proc_list[cur_device + 1] is not None:
            proc_list[cur_device + 1].join()
        utils.clean()

        if args.ft_epochs > 0:
            st = time.time()
            position_ids = position_ids.to(cur_device)
            attention_mask = attention_mask.to(cur_device)
            model.model.layers[i].to(cur_device)
            for j in range(args.devset_size // args.batch_size):
                orig_emb_cache[cur_device + 1][
                    args.batch_size * j : args.batch_size * (j + 1)] = \
                    model.model.layers[i](
                        orig_emb_cache[cur_device][
                            args.batch_size * j : args.batch_size * (j + 1)].to(cur_device),
                        position_ids=position_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_attentions=False)[0].cpu()
            model.model.layers[i].cpu()
            orig_msv = orig_emb_cache[cur_device].float().norm(
            )**2 / orig_emb_cache[cur_device].numel()
            target_msv = orig_emb_cache[cur_device + 1].float().norm(
            )**2 / orig_emb_cache[cur_device + 1].numel()
            position_ids = position_ids.cpu()
            attention_mask = attention_mask.cpu()
            utils.clean()
            glog.info(
                'computed original embedding for layer {} in {}s, pre msv {}, post msv {}'
                .format(i,
                        time.time() - st, orig_msv, target_msv))

        proc_list[cur_device] = mp.Process(target=quantize_llama_layer,
                                           args=(
                                               model.model.layers[i],
                                               i,
                                               cb,
                                               args,
                                               cur_device,
                                               orig_emb_cache[cur_device],
                                               orig_emb_cache[cur_device + 1],
                                               all_config['model_config'],
                                           ))
        proc_list[cur_device].start()

        cur_device = (cur_device + 1) % nproc

    for p in proc_list:
        p.join()


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    mp.set_start_method('spawn')
    mp.set_sharing_strategy('file_system')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.save_path, exist_ok=True)
    main(args)
