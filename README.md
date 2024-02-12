# QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks [[arXiv]](https://arxiv.org/abs/2402.04396)
This repository contains the official code for **QuIP#**, a weight-only post-training quantization method that achieves state-of-the-art performance in extreme compression ($\le 4$ bits per weight) regimes.
QuIP# improves the incoherence processing of [QuIP](https://openreview.net/pdf?id=xrk9g5vcXR) by using the randomized Hadamrd transform (RHT). 
QuIP# also introduces lattice codebooks based on the $E_8$ lattice and a fine-tuning scheme to further improve quantization quality.
With QuIP#, 3 bit models scale better than theoretically lossless 4 bit models, a previously unseen result.

We provide a full suite of 2, 3, and 4 bit Llama models quantized using QuIP# [here](https://huggingface.co/relaxml).
This codebase contains code that allows users to quantize and deploy their own models as well as CUDA kernels that accelerate inference for QuIP# models.
**Please open a GitHub ticket if you have any questions about the code or QuIP# in general.**

<img src="docs/img/quip.PNG" width="500">

| Method    | Precision | Wiki $\downarrow$ | C4 (`c4_new`) $\downarrow$  | ArcE $\uparrow$  | PiQA $\uparrow$  |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| Native    | 16 bit    |   3.120   |   5.533   |   0.597   |   0.809   |
| OPTQ      | 3 bit     |   4.577   |   6.838   |   0.544   | 0.786 |
| OPTQ      | 2 bit     |  109.820  |   62.692  |   0.253   |   0.505   |
| QuIP      | 2 bit     |   5.574   |   8.268   |   0.544   |   0.751   |
| **QuIP#** | **2 bit** | **3.907** | **6.248** | **0.591** | **0.794** |

## Latest Updates

- We released a preprint of QuIP# [here](https://arxiv.org/abs/2402.04396). The old blog post has not been updated so do not refer to that.
- The latest version of this codebase has been significantly revamped and cleaned up. It should be significantly easier to quantize new models. Make sure to pull from the repo to get the latest improvements to QuIP#. Better documentation will come soon.
- QuIP# now includes a fine-tuning algorithm that further improves quantization quality. The Llama models on the HF repo have been updated with new models quantized with fine-tuning. These models are fully compatible with the old code. The Openhermes and Mistral models have not been requantized with fine-tuning; we will update those in the near future.

## Installation

- Clone the repo
- Install the requirements via `pip install -r requirements.txt`. You may want to use the official pytorch commands to get the CUDA versions.
- Build and install the matmul CUDA kernels. (`cd quiptools && python setup.py install && cd ../`)

## Quantization

Example quantization scripts for the Llama family of models are located in `quantize_llama`. Follow these scripts to use QuIP# on other architectures. Within `quantize_llama`:
- `hessian_offline_llama.py` contains code to generate model Hessians. **Hessian calculation uses a `fp64` accumulator for numerical accuracy. Running this script on a device with slow `fp64` capabilities will take longer.** The HF repo includes pregenerated Hessians for a variety of models.
    - `--batch_size` Batch size per GPU. Tune so you don't run out of memory.
    - `--devset_size` Size of devset to use for Hessian generation.
    - `--ctx_size` Context size (sequence length) to use for Hessian generation.
    - `--base_model` Full precision HF model.
- `quantize_finetune_llama.py` contains code to quantize llama with fine-tuning ("fine-tuning during quantization" in the paper).
    - To reproduce earlier QuIP# results without fine-tuning, pass `--ft_epochs 0`
    - `--save_path` Output path.
    - `--base_model` Full precision HF model. For Llama 1, we provide weights at `relaxml/Llama-1-<7,13,30,65>b-hf`.
    - `--hessian_path` Offline Hessians. We provide precomputed Hessians at repo_id's `relaxml/Hessians*-<n>`. These Hessians were computed with `n` samples and the context length and attention mask used to train the original model. To download them, run `python scripts/download_hf.py --folder_path <local path to save Hessians> --repo_id <repo_id> --read_token <huggingface read token>`.
    - `--codebook` Codebook. Use `E8P12` for 2 bits, `E8P12RVQ3B` for 3 bits, and `E8P12RVQ4B` for 4 bits.
    - `--scale_override` and `--resid_scale_override`. Post-incoherence processing scale overrides. We suggest using 0.9 for `E8P12` and the default scales for 3 and 4 bit models. You may want to manually tune these for your specific model.
    - `--ft*` Various fine tuning arguments. `--ft_grad_ckpt` turns on gradient checkpointing and `--ft_train_mode` manifests the full quantized matrix during fine-tuning. We recommend turning `--ft_train_mode` on if you have enough memory since it makes training go faster.
- `finetune_e2e_llama.py` tunes the sign vectors (SU/SV), layernorms, and language model head of a given model (the second fine-tuning step in the paper). The arguments are similar to `quantize_finetune_llama.py`. You will need to convert the output of that script to a Hf model with `hfize_llama.py` before running this script. The HF-ized model should be passed in through `--hf_path`.
- `hfize_llama.py` converts a quantized model to the HF format. 

### I want to quantize a non-Llama architecture model, what do I do?

The scripts in `quantize_llama` are written with the Llama architecture in mind.
However, QuIP# is adaptable to any architecture with linear layers. 
To use QuIP# on a new architecture, identify the relevant linear layers and update the scripts in `quantize_llama`.
Feel free to open a GitHub issue if you run into issues.
    
## Evaluation

`eval` contains evaluation scripts. These scripts may need `CUDA_VISIBLE_DEVICES=0` if you run into CUDA errors due to how HF accelerate works. 
- `eval_ppl.py` calculates perplexity on Wikitext2 and C4.
- `eval_zeroshot.py` calculates performance on zeroshot tasks.
- `eval_speed.py` times the forward pass for one token.

## Text Generation

`eval/interactive_gen.py` contains a very simple interactive generation script. 
This script is very rudimentary and you may want to write your own - all it does is call HF's `.generate()` function.
**HF generate does not currently work with CUDA graphs. Thus, this script will be very slow since most of the time is spent on kernel launches. We expect better support for CUDA graphs with HF in transformers 4.38, which should come out in less than a month.**

## Model Zoo
We provide quantized models available on HF.
To use them, pass the given HF repo_id to `--hf_path`.
The 3 bit models are currently significantly slower than the 2 and 4 bit models during generation since we have not written an optimized matvec CUDA kernel for them yet.

| Lattice Codebook | Base Model  | Weight Bits | HF repo_id |
|:----------------:|:-----------|:-----------:|:----------------|
| E8P 2 Bit        | Llama 2 70b | 2           | [`relaxml/Llama-2-70b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-70b-E8P-2Bit) |
|                  | Llama 2 70b chat| 2       | [`relaxml/Llama-2-70b-chat-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-70b-chat-E8P-2Bit) |
|                  | Llama 2 13b | 2           | [`relaxml/Llama-2-13b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-13b-E8P-2Bit) |
|                  | Llama 2 13b chat| 2       | [`relaxml/Llama-2-13b-chat-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-13b-chat-E8P-2Bit) |
|                  | Llama 2 7b  | 2           | [`relaxml/Llama-2-7b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-7b-E8P-2Bit)   |
|                  | Llama 2 7b chat| 2        | [`relaxml/Llama-2-7b-chat-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-7b-chat-E8P-2Bit) |
|                  | Llama 1 65b | 2           | [`relaxml/Llama-1-65b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-1-65b-E8P-2Bit) |
|                  | Llama 1 30b | 2           | [`relaxml/Llama-1-30b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-1-30b-E8P-2Bit) |
|                  | Llama 1 13b | 2           | [`relaxml/Llama-1-13b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-1-13b-E8P-2Bit) |
|                  | Llama 1 7b  | 2           | [`relaxml/Llama-1-7b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-1-7b-E8P-2Bit)   |
|		   | Mistral 7b (non fine-tuned) | 2	       | [`relaxml/Mistral-7b-E8P-2Bit`](https://huggingface.co/relaxml/Mistral-7b-E8P-2Bit)   |
|		   | OpenHermes 2.5 (non fine-tuned) | 2	       | [`relaxml/Openhermes-7b-E8P-2Bit`](https://huggingface.co/relaxml/Openhermes-7b-E8P-2Bit)   |
| E8P RVQ 3 Bit    | Llama 2 70b | 3           | [`relaxml/Llama-2-70b-E8PRVQ-3Bit`](https://huggingface.co/relaxml/Llama-2-70b-E8PRVQ-3Bit) |
|                  | Llama 2 70b chat| 3       | [`relaxml/Llama-2-70b-chat-E8PRVQ-3Bit`](https://huggingface.co/relaxml/Llama-2-70b-chat-E8PRVQ-3Bit) |
|                  | Llama 2 13b | 3           | [`relaxml/Llama-2-13b-E8PRVQ-3Bit`](https://huggingface.co/relaxml/Llama-2-13b-E8PRVQ-3Bit) |
|                  | Llama 2 13b chat| 3       | [`relaxml/Llama-2-13b-chat-E8PRVQ-3Bit`](https://huggingface.co/relaxml/Llama-2-13b-chat-E8PRVQ-3Bit) |
|                  | Llama 2 7b  | 3           | [`relaxml/Llama-2-7b-E8PRVQ-3Bit`](https://huggingface.co/relaxml/Llama-2-7b-E8PRVQ-3Bit)   |
|                  | Llama 2 7b chat| 3        | [`relaxml/Llama-2-7b-chat-E8PRVQ-3Bit`](https://huggingface.co/relaxml/Llama-2-7b-chat-E8PRVQ-3Bit) |
|                  | Llama 1 65b | 3           | [`relaxml/Llama-1-65b-E8PRVQ-3Bit`](https://huggingface.co/relaxml/Llama-1-65b-E8PRVQ-3Bit) |
|                  | Llama 1 30b | 3           | [`relaxml/Llama-1-30b-E8PRVQ-3Bit`](https://huggingface.co/relaxml/Llama-1-30b-E8PRVQ-3Bit) |
|                  | Llama 1 13b | 3           | [`relaxml/Llama-1-13b-E8PRVQ-3Bit`](https://huggingface.co/relaxml/Llama-1-13b-E8PRVQ-3Bit) |
|                  | Llama 1 7b  | 3           | [`relaxml/Llama-1-7b-E8PRVQ-3Bit`](https://huggingface.co/relaxml/Llama-1-7b-E8PRVQ-3Bit)   |
|		   | Mistral 7b (non fine-tuned) | 3	       | [`relaxml/Mistral-7b-E8PRVQ-3Bit`](https://huggingface.co/relaxml/Mistral-7b-E8PRVQ-3Bit)   |
|		   | OpenHermes 2.5 (non fine-tuned) | 3	       | [`relaxml/Openhermes-7b-E8PRVQ-3Bit`](https://huggingface.co/relaxml/Openhermes-7b-E8PRVQ-3Bit)   |
| E8P RVQ 4 Bit    | Llama 2 70b | 4           | [`relaxml/Llama-2-70b-E8PRVQ-4Bit`](https://huggingface.co/relaxml/Llama-2-70b-E8PRVQ-4Bit) |
|                  | Llama 2 70b chat| 4       | [`relaxml/Llama-2-70b-chat-E8PRVQ-4Bit`](https://huggingface.co/relaxml/Llama-2-70b-chat-E8PRVQ-4Bit) |
|                  | Llama 2 13b | 4           | [`relaxml/Llama-2-13b-E8PRVQ-4Bit`](https://huggingface.co/relaxml/Llama-2-13b-E8PRVQ-4Bit) |
|                  | Llama 2 13b chat| 4       | [`relaxml/Llama-2-13b-chat-E8PRVQ-4Bit`](https://huggingface.co/relaxml/Llama-2-13b-chat-E8PRVQ-4Bit) |
|                  | Llama 2 7b  | 4           | [`relaxml/Llama-2-7b-E8PRVQ-4Bit`](https://huggingface.co/relaxml/Llama-2-7b-E8PRVQ-4Bit)   |
|                  | Llama 2 7b chat| 4        | [`relaxml/Llama-2-7b-chat-E8PRVQ-4Bit`](https://huggingface.co/relaxml/Llama-2-7b-chat-E8PRVQ-4Bit) |
|                  | Llama 1 65b | 4           | [`relaxml/Llama-1-65b-E8PRVQ-4Bit`](https://huggingface.co/relaxml/Llama-1-65b-E8PRVQ-4Bit) |
|                  | Llama 1 30b | 4           | [`relaxml/Llama-1-30b-E8PRVQ-4Bit`](https://huggingface.co/relaxml/Llama-1-30b-E8PRVQ-4Bit) |
|                  | Llama 1 13b | 4           | [`relaxml/Llama-1-13b-E8PRVQ-4Bit`](https://huggingface.co/relaxml/Llama-1-13b-E8PRVQ-4Bit) |
|                  | Llama 1 7b  | 4           | [`relaxml/Llama-1-7b-E8PRVQ-4Bit`](https://huggingface.co/relaxml/Llama-1-7b-E8PRVQ-4Bit)   |
|		   | Mistral 7b (non fine-tuned) | 4	       | [`relaxml/Mistral-7b-E8PRVQ-4Bit`](https://huggingface.co/relaxml/Mistral-7b-E8PRVQ-4Bit)   |
|		   | OpenHermes 2.5 (non fine-tuned) | 4	       | [`relaxml/Openhermes-7b-E8PRVQ-4Bit`](https://huggingface.co/relaxml/Openhermes-7b-E8PRVQ-4Bit)   |


## CUDA Graphs

We provide a wrapper class that integrates our models with CUDA graphs in `model/graph_wrapper.py`.
Currently, the torch CUDA graph implementation does not work with HF's `.generate()` function, but model calls with static input and output sizes can utilize the CUDA graph wrapper for better performance.
Most of our evaluation scripts use the graph wrapper by default unless the `--no_use_cuda_graph` flag is passed in.

## Other

Use of Llama models is governed by the Meta license available [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).
Use of Mistral models is governed by the Apache 2.0 license.
Use of this code is governed by the GNU GPL v3 license.

If you found this work useful, please consider citing
```
@misc{tseng2024quip,
      title={QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks}, 
      author={Albert Tseng and Jerry Chee and Qingyao Sun and Volodymyr Kuleshov and Christopher De Sa},
      year={2024},
      eprint={2402.04396},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
