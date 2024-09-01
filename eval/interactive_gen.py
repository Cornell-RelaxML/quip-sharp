# Most of this code is from https://github.com/chu-tianxiang/QuIP-for-all

import time
from typing import Optional

import torch
from transformers import AutoTokenizer, StaticCache

from lib.utils.unsafe_import import model_from_hf_path
torch.set_grad_enabled(False)
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

@torch.no_grad()
def decode_one_tokens(model, cur_token, cache_position):
    logits = model(cur_token, cache_position=cache_position, return_dict=False, use_cache=True)[0]
    new_token = sample(logits,temperature=0.6, top_k=5)[0]
    return new_token


@torch.no_grad()
def generate(model, tokenizer, text, max_new_tokens, top_k, callback):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    batch_size, seq_length = inputs["input_ids"].shape
    cache_position = torch.arange(seq_length, device=model.device)
    generated_ids = torch.zeros(
        batch_size, seq_length + max_new_tokens, dtype=torch.int, device=model.device
    )
    generated_ids[:, cache_position] = inputs["input_ids"].to(model.device).to(torch.int)

    logits = model(**inputs, cache_position=cache_position, return_dict=False, use_cache=True)[0]
    next_token, _ = sample(logits, top_k=top_k)
    generated_ids[:, seq_length] = next_token
    callback(next_token)

    cache_position = torch.tensor([seq_length + 1], device=model.device)
    decode_time = time.time()
    for _ in range(1, max_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=True):
            next_token = decode_one_tokens(model, next_token.clone(), cache_position)
        generated_ids[:, cache_position] = next_token.int()
        callback(next_token)
        cache_position += 1
    torch.cuda.synchronize()
    decode_time = time.time() - decode_time

    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return generated_ids, text, max_new_tokens/decode_time


def main(hf_path, compile, interactive, num_samples, max_tokens, top_k):
    device = "cuda"
    model, model_str = model_from_hf_path(
        hf_path,
        use_cuda_graph=False,
        device_map='cuda:0')

    tokenizer = AutoTokenizer.from_pretrained(model_str)
    tokenizer.pad_token = tokenizer.eos_token
    model._setup_cache(StaticCache, 1, max_cache_len=2048)
    
    if compile:
        print('Capturing CUDA graphs, may take some time.')
        global decode_one_tokens
        decode_one_tokens = torch.compile(decode_one_tokens, mode="max-autotune", fullgraph=True)
        text = "Test"
        callback = lambda x : x
        ids, text, _ = generate(model, tokenizer, text, max_tokens, top_k, callback)

        
    while True:
        prompt = input("What is your prompt? ")
        if prompt == 'quit':
            exit()

        if tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            text = prompt
        buffer = []
        period_id = tokenizer.encode('.')[-1]
        done_generating = False
        def callback(x):
            nonlocal done_generating
            if done_generating:
                return
            buffer.append(tokenizer.decode([period_id] + x[0].tolist())[1:])
            if x[0].item() == tokenizer.eos_token_id:
                done_generating = True
            if len(buffer) == 4 or done_generating:
                print(''.join(buffer), end='', flush=True)
                buffer.clear()
        if not interactive:
            callback = lambda x : x
        ids, text, decode_tps = generate(model, tokenizer, text, max_tokens, top_k, callback)
        if not interactive:
            print(text)
        else:
            print()
        print(f"Decoding throughput: {decode_tps:.02f} tokens/sec")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--hf_path', type=str, help="Path to checkpoint")
    parser.add_argument('--streaming', action='store_true', help='Whether to launch in stream mode')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=512, help='Maximum number of new tokens.')
    parser.add_argument('--top_k', type=int, default=32, help='Top-k for sampling.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')

    args = parser.parse_args()
    main(args.hf_path, args.compile, args.streaming, args.num_samples, args.max_new_tokens, args.top_k)
