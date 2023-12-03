import os
import json
import argparse
import torch
import datasets
from transformers import AutoTokenizer
import random
import glog

from lib.utils import LMEvalAdaptor
from lib.utils.unsafe_import import model_from_hf_path
from lm_eval import evaluator, tasks

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--hf_path', default='hfized/quantized_hada_70b', type=str)
parser.add_argument('--batch_size', type=int, default=1, help='batch size')
parser.add_argument("--tasks", type=str)
parser.add_argument("--output_path", default=None, type=str)
parser.add_argument('--num_fewshot', type=int, default=0)
parser.add_argument('--no_use_cuda_graph', action='store_true')
parser.add_argument('--no_use_flash_attn', action='store_true')


def main(args):
    model, model_str = model_from_hf_path(args.hf_path,
                                          use_cuda_graph=False,
                                          use_flash_attn=not args.no_use_flash_attn)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    glog.info('loaded model!')
    tokenizer.pad_token = tokenizer.eos_token

    task_names = args.tasks.split(",")

    lm_eval_model = LMEvalAdaptor(model_str, model, tokenizer, args.batch_size)
    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        batch_size=args.batch_size,
        no_cache=True,
        num_fewshot=args.num_fewshot,
    )

    print(evaluator.make_table(results))

    if args.output_path is not None:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        # otherwise cannot save
        results["config"]["model"] = args.hf_path
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == '__main__':
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
