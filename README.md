# QuIP#: [QuIP](https://github.com/jerry-chee/QuIP) with Lattice Codebooks
This repository contains the official code for **QuIP#**, a weights-only quantization method that is able to achieve near fp16 performance using only 2 bits per weight.
QuIP# combines lattice codebooks with incoherence processing to create state-of-the-art 2 bit quantized models.
We provide a full suite of 2 bit Llama models quantized using QuIP# as well as other Llama-architecture models (e.g. Mistral).
We also provide a full codebase that allows users to quantize and deploy their own models as well as CUDA kernels that accelerate inference for QuIP# models.

| Method    | Precision | Wiki $\downarrow$ | C4 $\downarrow$  | ArcE $\uparrow$  | PiQA $\uparrow$  |
|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| Native    | 16 bit    |   3.120   |   5.533   |   0.597   |   0.809   |
| OPTQ      | 3 bit     |   4.577   |   6.838   |   0.544   | **0.786** |
| OPTQ      | 2 bit     |  109.820  |   62.692  |   0.253   |   0.505   |
| QuIP      | 2 bit     |   5.574   |   8.268   |   0.544   |   0.751   |
| **QuIP#** | **2 bit** | **4.156** | **6.545** | **0.595** |   0.785   |

Quantization results on Llama 2 70B. QuIP# achieves near-native performance at 2 bits, outperforming all other presented baselines.

## ☞ Read more about QuIP# and how it works [here](https://cornell-relaxml.github.io/quip-sharp/)!

## News

- We have "deprecated" the 2 bit D4 quantized models as they perform worse than 2 bit E8P models and are slower to run. The code to quantize and run D4 models is still in the codebase, but the D4 models have been removed from HF and we are no longer actively supporting them.
- We recently added 2 and 4 bit quantized versions of [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-v0.1) and [OpenHermes 2.5](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B). See the Model Zoo section for more details.
- **The 4 bit models have been replaced by new bit-packed models that end with the `-Packed` suffix. The old models have been deprecated, removed, and do not work with the current code (and vice versa). Make sure to pull the latest code to run the 4 bit models.**

## Installation

- Clone the repo
- Install the requirements via `pip install -r requirements.txt`. You may want to use the official pytorch commands to get the CUDA versions.
- Build and install the matmul CUDA kernels. (`cd quiptools && python setup.py install && cd ../`)

## Quantization

- To quantize a Llama architecture (q/k/v/o/up/gate/down) model: `python quantize_llama.py --<FLAGS>`. The primary flags are as follows. See the arg list for the remaining flags.
    - `--save_path <output path>`.
    - `--base_model <Hugging Face (HF) model card or local path>`. 
    For Llama 1, we provide weights at `relaxml/Llama-1-<7,13,30,65>b-hf`. For other models, use model cards from HF.
    - `--hessian_path <path to precomputed Hessians>`. 
    We provide precomputed Hessians at repo_id's `relaxml/Hessians*-<n>`. These Hessians were computed with `n` samples and the context length and attention mask used to train the original model. To download them, run `python scripts/download_hf.py --folder_path <local path to save Hessians> --repo_id <repo_id> --read_token <huggingface read token>`.
    - `--codebook <codebook argument>`. 
    We recommend using the 2 bit E8P codebook with `E8P12`. This codebook gives the best quantization at 2 bits. Other options are the 2 bit `D4` codebook and the 4 bit Half Integer grid `HI4B1C`. See our blog post for details on the codebooks.
    - `--scale_override <quantization scale parameter>`. 
    We suggest the following scale parameters for each codebook: `{E8P12: 0.9, D4: 1.1, HI4B1C: 2.7}`, however you may want to play around with scales if quantizing your own models. 
- To convert a quantized model to the HF format: `CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path <output path of quantize_llama.py> --hf_output_path <path to save HF version>`
- To generate your own Hessians for a Llama architecture model: `python hessian_offline_llama --<FLAGS>`. The primary flags are as follows. See the arg list for the remaining flags. **Hessian calculation uses a `fp64` accumulator for numerical accuracy. Running this script on a device with slow `fp64` capabilities will take longer.**
    - `--batch_size` Batch size per GPU. Tune so you don't run out of memory.
    - `--devset_size` Size of devset to use for Hessian generation.
    - `--ctx_size` Context size (sequence length) to use for Hessian generation.
    - `--base_model` Same as in `quantize_llama.py`.

### I want to quantize a non-Llama architecture model, what do I do?

Currently, `hessian_offline_llama.py`, `quantize_llama.py`, and `hfize_llama.py` are written for the Llama architecture. However, the only "special" things they do are identify the relevant `nn.Linear` layers that need to be quantized (q/k/v/o/up/gate/down), inject Hessian hooks, and quantize them. 
If you want to quantize a non-Llama architecture model, you will need to find the relevant `nn.Linear` files and make your own hessian_offline/quantize/hfize files. This should be pretty straightforward and feel free to open a GitHub ticket if you run into any issues.
You will also need copy `modeling_<architecture>.py` from the HF source into the `models/` folder and replace the relevant `nn.Linear` layers with `QuantizedLinear` layers (see how `models/llama.py` does it).
Our current `quantize_llama.py` implementation fuses the q/k/v layers and the up/gate layers for increased speed since they share the same Hessians. However, this is not a requirement and you can also quantize those layers individually.

    
## Evaluation

See our blog post for a full set of results.
- Perplexity on Wikitext2 and C4: `CUDA_VISIBLE_DEVICES=0 python eval_ppl.py --hf_path <HF version path>`
- Zero shot tasks: `CUDA_VISIBLE_DEVICES=0 python eval_zeroshot.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size <batch size> --hf_path <HF version path>`
- Timing test for forward pass of one token: `CUDA_VISIBLE_DEVICES=0 python gen_speed.py --hf_path <HF version path> --batch_size <batch_size>`.

*The `CUDA_VISIBLE_DEVICES` environmental variable is only needed if you get CUDA errors from running on more GPUs than needed to fit the model. This is an artifact of HF accelerate.*

## Text Generation

To use our models as part of an interactive generation script, run `CUDA_VISIBLE_DEVICES=0 python interactive_gen.py --hf_path <HF version path> --max_length <max generation length>`.
`interactive_gen.py` is very rudimentary and you may want to write your own.
All it does is call HF's `.generate()` function.

## Model Zoo
We provide quantized models available on HF.
To use them, pass the given HF repo_id to `--hf_path`.
We recommend using the `E8P` codebook which quantizes to 2 bits per weight, which gives the best quantization at 2 bits.
See our blogpost for details on the codebooks.

| Lattice Codebook | Base Model  | Weight Bits | HF repo_id |
|:----------------:|:-----------|:-----------:|:----------------|
| E8P (recommended)| Llama 2 70b | 2           | [`relaxml/Llama-2-70b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-70b-E8P-2Bit) |
|                  | Llama 2 70b chat| 2       | [`relaxml/Llama-2-70b-chat-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-70b-chat-E8P-2Bit) |
|                  | Llama 2 13b | 2           | [`relaxml/Llama-2-13b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-13b-E8P-2Bit) |
|                  | Llama 2 13b chat| 2       | [`relaxml/Llama-2-13b-chat-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-13b-chat-E8P-2Bit) |
|                  | Llama 2 7b  | 2           | [`relaxml/Llama-2-7b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-7b-E8P-2Bit)   |
|                  | Llama 2 7b chat| 2       | [`relaxml/Llama-2-7b-chat-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-7b-chat-E8P-2Bit) |
|                  | Llama 1 65b | 2           | [`relaxml/Llama-1-65b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-1-65b-E8P-2Bit) |
|                  | Llama 1 30b | 2           | [`relaxml/Llama-1-30b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-1-30b-E8P-2Bit) |
|                  | Llama 1 13b | 2           | [`relaxml/Llama-1-13b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-1-13b-E8P-2Bit) |
|                  | Llama 1 7b  | 2           | [`relaxml/Llama-1-7b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-1-7b-E8P-2Bit)   |
|		   | Mistral 7b  | 2	       | [`relaxml/Mistral-7b-E8P-2Bit`](https://huggingface.co/relaxml/Mistral-7b-E8P-2Bit)   |
|		   | OpenHermes 2.5 | 2	       | [`relaxml/Openhermes-7b-E8P-2Bit`](https://huggingface.co/relaxml/Openhermes-7b-E8P-2Bit)   |
| HI               | Llama 2 70b | 4           | [`relaxml/Llama-2-70b-HI-4Bit-Packed`](https://huggingface.co/relaxml/Llama-2-70b-HI-4Bit-Packed) |
|                  | Llama 2 13b | 4           | [`relaxml/Llama-2-13b-HI-4Bit-Packed`](https://huggingface.co/relaxml/Llama-2-13b-HI-4Bit-Packed) |
|                  | Llama 2 7b  | 4           | [`relaxml/Llama-2-7b-HI-4Bit-Packed`](https://huggingface.co/relaxml/Llama-2-7b-HI-4Bit-Packed)   |
|                  | Llama 1 65b | 4           | [`relaxml/Llama-1-65b-HI-4Bit-Packed`](https://huggingface.co/relaxml/Llama-1-65b-HI-4Bit-Packed) |
|                  | Llama 1 30b | 4           | [`relaxml/Llama-1-30b-HI-4Bit-Packed`](https://huggingface.co/relaxml/Llama-1-30b-HI-4Bit-Packed) |
|                  | Llama 1 13b | 4           | [`relaxml/Llama-1-13b-HI-4Bit-Packed`](https://huggingface.co/relaxml/Llama-1-13b-HI-4Bit-Packed) |
|                  | Llama 1 7b  | 4           | [`relaxml/Llama-1-7b-HI-4Bit-Packed`](https://huggingface.co/relaxml/Llama-1-7b-HI-4Bit-Packed)   |
|		   | Mistral 7b  | 4	       | [`relaxml/Mistral-7b-HI-4Bit-Packed`](https://huggingface.co/relaxml/Mistral-7b-HI-4Bit-Packed)   |
|		   | OpenHermes 2.5 | 4	       | [`relaxml/Openhermes-7b-HI-4Bit-Packed`](https://huggingface.co/relaxml/Openhermes-7b-HI-4Bit-Packed)   |


## CUDA Graphs

We provide a wrapper class that integrates our models with CUDA graphs in `model/graph_wrapper.py`.
Currently, the torch CUDA graph implementation does not work with HF's `.generate()` function, but model calls with static input and output sizes can utilize the CUDA graph wrapper for better performance.
Most of our evaluation scripts use the graph wrapper by default unless the `--no_use_cuda_graph` flag is passed in.

## Other

Use of Llama models is governed by the Meta license available [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).
Use of Mistral models is governed by the Apache 2.0 license.
Use of this code is governed by the GNU GPL v3 license.
