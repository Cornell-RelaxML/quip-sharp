import os
import math
import json
import argparse
import torch
import datasets
from lib.utils import gptq_data_utils
from lib.utils.unsafe_import import model_from_hf_path
import random
import glog

from tqdm import tqdm
import sys # use sys argv to avoid arg conflict
from modules.llamacpp_hf import LlamacppHF
from modules import shared
shared.args.cfg_cache = True
shared.args.logits_all = True

torch.set_grad_enabled(False)

def main():
    datasets = ['wikitext2', 'c4']
    model = LlamacppHF.from_pretrained(shared.args.model_dir)

    # model str gets tokenizer
    if 'v2' in shared.args.model_dir:
        seqlen = 4096
        model_str = 'meta-llama/Llama-2-7b-hf'
    else:
        seqlen = 2048
        model_str = '/mnt/desa_data/meta_llama1/huggingface_7B'
        
    for dataset in datasets:
        input_tok = gptq_data_utils.get_test_tokens(dataset,
                                                    seed=0,
                                                    seqlen=seqlen,
                                                    model=model_str)
        nsamples = input_tok.numel() // seqlen
        input_tok = input_tok[0, :(seqlen * nsamples)].view(nsamples, seqlen)
        loss_fct = torch.nn.CrossEntropyLoss().cuda()
        acc_loss = 0.0
        progress = tqdm(range(nsamples))
        for ii in progress:
            input = input_tok[ii, :].cuda().view(1, -1)
            output = model(input_ids=input,
                           use_cache=False,
                           output_hidden_states=True,
                           output_attentions=False)
            shift_logits = torch.tensor(output[:-1]).cuda()
            shift_labels = torch.tensor(input[0][1:]).cuda()
            loss = loss_fct(shift_logits, shift_labels)
            acc_loss += loss.item()
            progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

        avg_loss = acc_loss / nsamples

        ppl = torch.exp(torch.tensor(avg_loss)).item()
        glog.info(f'{dataset} perplexity: {ppl}')


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    random.seed(0)
    torch.random.manual_seed(0)
    main()
