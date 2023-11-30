# QuIP#: [QuIP](https://github.com/jerry-chee/QuIP) with Lattice Codebooks
This repository contains the official code for **QuIP#**, a weights-only quantization method that is able to achieve near fp16 performance using only 2 bits per weight.
QuIP# combines lattice codebooks with incoherence processing to create state-of-the-art 2 bit quantized models.
We provide a full suite of 2 bit Llama 1 and 2 models quantized using QuIP#, including Llama 2 chat models.
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

## Installation

- Clone the repo
- Install the requirements via `pip install -r requirements.txt`. You may want to use the official pytorch commands to get the CUDA versions.
- Build and install the hadamard and matmul kernels. (`cd hadamard_cuda && python setup.py install && cd ../quiptools && python setup.py install && cd ../`)

## Quantization

- To quantize: `python quantize_llama.py --<FLAGS>`. The primary flags are as follows. See the arg list for the remaining flags.
    - `--save_path <output path>`.
    - `--base_model <Llama 1 or 2 model>`. 
    For Llama 1, we provide weights at `relaxml/Llama-1-<7,13,30,65>b-hf`. For Llama 2, use the officially provided weights at https://huggingface.co/meta-llama. 
    - `--hessian_path <path to precomputed hessians>`. 
    For Llama 1 we provide precomputed hessians at repo_id's `relaxml/Hessians-Llama-1-<7,13,30,65>b-6144`. For Llama 2 we provide precomputed hessians at repo_id's `relaxml/Hessians-Llama-2-<7,13,70>b-6144`. To download them, run `python scripts/download_hf.py --folder_path <local path to save hessians> --repo_id <repo_id> --read_token <huggingface read token>`.
    - `--codebook <codebook argument>`. 
    We recommend using the 2 bit E8P codebook with `E8P12`. This codebook gives the best quantization at 2 bits. Other options are the `D4` codebook at 2 bits, and the 4 bit Half Integer 1 Column grid with `HI4B1C`. See our blogpost for details on the codebooks.
    - `--scale_override <quantization scale parameter>`. 
    We suggest the following scale parameters for each codebook: `{E8P12: 0.9, D4: 1.1, HI4B1C: 2.7}`. 
- To convert a quantized model to the Hugging Face (HF) format: `CUDA_VISIBLE_DEVICES=0 python hfize_llama.py --quantized_path <output path of quantize_llama.py> --hf_output_path <path to save HF version>`

## Evaluation

See our blog post for a full set of results.
- Perplexity on Wikitext2 and C4: `CUDA_VISIBLE_DEVICES=0 python ppl_llama.py --hf_path <HF version path>`
- Zero shot tasks: `CUDA_VISIBLE_DEVICES=0 python eval_llama.py --tasks arc_challenge,arc_easy,boolq,piqa,winogrande --batch_size <batch size> --hf_path <HF version path>`
- Timing test for forward pass of one token: `CUDA_VISIBLE_DEVICES=0 python gen_speed.py --hf_path <HF version path> --batch_size <batch_size>`.

*The `CUDA_VISIBLE_DEVICES` environmental variable is only needed if you get CUDA errors from running on more GPUs than needed to fit the model. This is an artifact from HF accelerate.*

## Text Generation

To use our models as part of an interactive generation script, run `CUDA_VISIBLE_DEVICES=0 python interactive_gen.py --hf_path <HF version path> --max_length <max generation length>`.

## Model Zoo
We provide Llama 1 and Llama 2 quantized models available on Hugging Face.
To use them, pass the given Hugging Face repo_id to `--hf_path`.
We recommend using the `E8P` codebook which quantizes to 2 bits per weight, which gives the best quantization at 2 bits.
Other options are the `D4` codebook at 2 bits, and the half-integer grid `HI4B1C` codebook at 4 bits.
See our blogpost for details on the codebooks.

| Lattice Codebook | Base Model  | Weight Bits | Hugging Face repo_id |
|:----------------:|:-----------|:-----------:|:----------------|
| E8P              | Llama 2 70b | 2           | [`relaxml/Llama-2-70b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-70b-E8P-2Bit) |
|                  | Llama 2 70b chat| 2       | [`relaxml/Llama-2-70b-chat-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-70b-chat-E8P-2Bit) |
|                  | Llama 2 13b | 2           | [`relaxml/Llama-2-13b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-13b-E8P-2Bit) |
|                  | Llama 2 13b chat| 2       | [`relaxml/Llama-2-13b-chat-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-13b-chat-E8P-2Bit) |
|                  | Llama 2 7b  | 2           | [`relaxml/Llama-2-7b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-7b-E8P-2Bit)   |
|                  | Llama 2 7b chat| 2       | [`relaxml/Llama-2-7b-chat-E8P-2Bit`](https://huggingface.co/relaxml/Llama-2-7b-chat-E8P-2Bit) |
|                  | Llama 1 65b | 2           | [`relaxml/Llama-1-65b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-1-65b-E8P-2Bit) |
|                  | Llama 1 30b | 2           | [`relaxml/Llama-1-30b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-1-30b-E8P-2Bit) |
|                  | Llama 1 13b | 2           | [`relaxml/Llama-1-13b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-1-13b-E8P-2Bit) |
|                  | Llama 1 7b  | 2           | [`relaxml/Llama-1-7b-E8P-2Bit`](https://huggingface.co/relaxml/Llama-1-7b-E8P-2Bit)   |
| D4               | Llama 2 70b | 2           | [`relaxml/Llama-2-70b-D4-2Bit`](https://huggingface.co/relaxml/Llama-2-70b-D4-2Bit) |
|                  | Llama 2 13b | 2           | [`relaxml/Llama-2-13b-D4-2Bit`](https://huggingface.co/relaxml/Llama-2-13b-D4-2Bit) |
|                  | Llama 2 7b  | 2           | [`relaxml/Llama-2-7b-D4-2Bit`](https://huggingface.co/relaxml/Llama-2-7b-D4-2Bit)   |
|                  | Llama 1 65b | 2           | [`relaxml/Llama-1-65b-D4-2Bit`](https://huggingface.co/relaxml/Llama-1-65b-D4-2Bit) |
|                  | Llama 1 30b | 2           | [`relaxml/Llama-1-30b-D4-2Bit`](https://huggingface.co/relaxml/Llama-1-30b-D4-2Bit) |
|                  | Llama 1 13b | 2           | [`relaxml/Llama-1-13b-D4-2Bit`](https://huggingface.co/relaxml/Llama-1-13b-D4-2Bit) |
|                  | Llama 1 7b  | 2           | [`relaxml/Llama-1-7b-D4-2Bit`](https://huggingface.co/relaxml/Llama-1-7b-D4-2Bit)   |
| HI               | Llama 2 70b | 4           | [`relaxml/Llama-2-70b-HI-4Bit`](https://huggingface.co/relaxml/Llama-2-70b-HI-4Bit) |
|                  | Llama 2 13b | 4           | [`relaxml/Llama-2-13b-HI-4Bit`](https://huggingface.co/relaxml/Llama-2-13b-HI-4Bit) |
|                  | Llama 2 7b  | 4           | [`relaxml/Llama-2-7b-HI-4Bit`](https://huggingface.co/relaxml/Llama-2-7b-HI-4Bit)   |
|                  | Llama 1 65b | 4           | [`relaxml/Llama-1-65b-HI-4Bit`](https://huggingface.co/relaxml/Llama-1-65b-HI-4Bit) |
|                  | Llama 1 30b | 4           | [`relaxml/Llama-1-30b-HI-4Bit`](https://huggingface.co/relaxml/Llama-1-30b-HI-4Bit) |
|                  | Llama 1 13b | 4           | [`relaxml/Llama-1-13b-HI-4Bit`](https://huggingface.co/relaxml/Llama-1-13b-HI-4Bit) |
|                  | Llama 1 7b  | 4           | [`relaxml/Llama-1-7b-HI-4Bit`](https://huggingface.co/relaxml/Llama-1-7b-HI-4Bit)   |


## CUDA Graphs

We provide a wrapper class that integrates our models with CUDA graphs in `model/graph_wrapper.py`.
Currently, the torch CUDA graph implementation does not work with Hugging Face's `.generate()` function, but model calls with static input and output sizes can utilize the CUDA graph wrapper for better performance.
Most of our evaluation scripts use the graph wrapper by default unless the `--no_use_cuda_graph` flag is passed in.

## Other

Use of Llama models is governed by the Meta license avaiable [here](https://ai.meta.com/resources/models-and-libraries/llama-downloads/).
Use of this code is governed by the GNU GPL v3 license.
