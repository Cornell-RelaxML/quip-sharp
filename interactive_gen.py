import argparse
import os
import glog
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import LlamaTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from lib.utils.unsafe_import import model_from_hf_path
import time

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser()
parser.add_argument('--hf_path', default='meta-llama/Llama-2-70b-hf', type=str)
parser.add_argument('--max_tokens', default=64, type=int)
parser.add_argument('--num_beams', default=4, type=int)
parser.add_argument('--no_use_flash_attn', action='store_true')


def main(args):
    model, model_str = model_from_hf_path(args.hf_path,
                                          use_cuda_graph=False,
                                          use_flash_attn=not args.no_use_flash_attn)
    tokenizer = LlamaTokenizer.from_pretrained(model_str)
    tokenizer.pad_token = tokenizer.eos_token

    print("Using {args.num_beams} beams to generate up to {args.max_tokens} tokens.")

    while True:
        print()
        prompt = input("Please enter your prompt or 'quit' (without quotes) to quit: ")
        if prompt == 'quit':
            return
        inputs = tokenizer(prompt, return_tensors='pt')
        outputs = model.generate(input_ids=inputs['input_ids'].cuda(),
                                 attention_mask=inputs['attention_mask'].cuda(),
                                 max_new_tokens=args.max_tokens,
                                 num_beams=args.num_beams,
                                 pad_token_id=tokenizer.eos_token_id,
                                 return_dict_in_generate=True).sequences[0]
        print()
        print('Model Output: ', tokenizer.decode(outputs, skip_special_tokens=True))


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.manual_seed(0)
    args = parser.parse_args()
    main(args)
