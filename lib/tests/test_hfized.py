import time
import torch
from transformers import LlamaTokenizer  #, LlamaForCausalLM
from model.llama import LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained('./llama_hada_d4_70b', torch_dtype="auto", low_cpu_mem_usage=True)
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-70b-hf")
# model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-70b-chat-hf",torch_dtype="auto",low_cpu_mem_usage=True)
# import latticed4

from torch.profiler import profile, record_function, ProfilerActivity

torch.set_grad_enabled(False)
torch.set_float32_matmul_precision('high')

# D4_CB = latticed4.build_D4_CB().cuda()

# def load_quip(save_name, D4_CB):
#     print(f"loading cached compressed layer from path \"{save_name}\"")
#     dict_loaded = torch.load(save_name)
#     A = dict_loaded['A'].to(torch.float32)
#     B = dict_loaded['B'].to(torch.float32)
#     U1 = dict_loaded['U1'].to(torch.float32)
#     U2 = dict_loaded['U2'].to(torch.float32)
#     V1 = dict_loaded['V1'].to(torch.float32)
#     V2 = dict_loaded['V2'].to(torch.float32)
#     Qidxs = dict_loaded['Qidxs'].to(torch.int64)
#     # Qscales = dict_loaded['Qscales'].to(torch.float32)
#     p1 = U1.shape[0]
#     p2 = U2.shape[0]
#     q1 = V1.shape[0]
#     q2 = V2.shape[0]
#     (m,n) = Qidxs.shape
#     hatWr = torch.zeros(m,4*n,device=A.device)
#     for k in range(n):
#         hatWr[:,(4*k):(4*(k+1))] = D4_CB[Qidxs[:,k],:] # * Qscales[k]
#     hatWr.add_(A @ B)
#     # U = torch.kron(U1,U2)
#     # V = torch.kron(V1,V2)
#     # hatW = V.t() @ (hatWr) @ U
#     hatW = torch.einsum("ijkl,ia,jb,kc,ld->abcd",hatWr.view(q1,q2,p1,p2), V1, V2, U1, U2).view(m,4*n)
#     return (hatW, dict_loaded)

# def to_gpu_hook(module,input):
#     module.cuda()
# def from_gpu_hook(module,input,output):
#     module.cpu()

# tmp_layers = model.model.layers
# model.model.layers = None
model.half()
model.cuda()

# model.model = torch.compile(model.model, backend="eager", dynamic=False)

# model.lm_head.to(torch.float32)
# model.model.layers = tmp_layers

# for (transformer_layer_index, transformer_layer) in enumerate(model.model.layers):
#     print(f"processing layer {transformer_layer_index}")

#     transformer_layer.register_forward_pre_hook(to_gpu_hook)
#     transformer_layer.register_forward_hook(from_gpu_hook)

#     (hatW, dict_loaded) = load_quip(f'quantized-reg0.01/{transformer_layer_index}_qkv.pt', D4_CB)

#     W_q = transformer_layer.self_attn.q_proj.weight
#     W_k = transformer_layer.self_attn.k_proj.weight
#     W_v = transformer_layer.self_attn.v_proj.weight

#     W_q_next = (hatW[0:(W_q.shape[0]),:]*dict_loaded['W_q_scale']).half()
#     W_k_next = (hatW[(W_q.shape[0]):(W_q.shape[0]+W_k.shape[0]),:]*dict_loaded['W_k_scale']).half()
#     W_v_next = (hatW[(W_q.shape[0]+W_k.shape[0]):(W_q.shape[0]+W_k.shape[0]+W_v.shape[0]),:]*dict_loaded['W_v_scale']).half()

#     W_q.copy_(W_q_next)
#     W_k.copy_(W_k_next)
#     W_v.copy_(W_v_next)

#     (hatW, dict_loaded) = load_quip(f'quantized-reg0.01/{transformer_layer_index}_o.pt', D4_CB)

#     transformer_layer.self_attn.o_proj.weight.copy_(hatW*dict_loaded['W_o_scale'])

#     (hatW, dict_loaded) = load_quip(f'quantized-reg0.01/{transformer_layer_index}_up.pt', D4_CB)

#     W_up = transformer_layer.mlp.up_proj.weight
#     W_gate = transformer_layer.mlp.gate_proj.weight

#     W_up_next = (hatW[0:(W_up.shape[0]),:]*dict_loaded['W_up_scale']).half()
#     W_gate_next = (hatW[(W_up.shape[0]):(W_up.shape[0]+W_gate.shape[0]),:]*dict_loaded['W_gate_scale']).half()

#     W_up.copy_(W_up_next)
#     W_gate.copy_(W_gate_next)

#     (hatW, dict_loaded) = load_quip(f'quantized-reg0.01/{transformer_layer_index}_down.pt', D4_CB)

#     transformer_layer.mlp.down_proj.weight.copy_(hatW*dict_loaded['W_down_scale'])

print("generating some text...")

# B_INST, E_INST = "[INST]", "[/INST]"
# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
# DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
#
# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. Always answer as helpfully as possible. If you don't know the answer to a question, please guess or make something up."""

prompts = [
    "[INST] What fun things can I do during my conference trip to Vienna? [/INST] ",
    "[INST] Write me a romantic Harry Potter fanfic about Draco Malfoy. [/INST] ",
    "[INST] Please give me a traditional recipe for the Chinese Dish called Tomato Egg. [/INST] ",
    "[INST] What is the best chess opening for a player rated around 1400? [/INST] ",
    "[INST] Please summarize Andrew Ng's opinions on AI.  [/INST] ",
    "[INST] Tell me the fairy tail of the mouse, the bird, and the sausage. [/INST] ",
    "[INST] In The Music Man, what was the problem with Pool? [/INST] ",
    "[INST] Please list all the UK prime ministers in order. [/INST] ",
    # "[INST] What do you think about the events in Washington DC on January 6th, 2021? [/INST]",
    # "[INST] Please tell me some good songs by The Killers to play at my wedding. [/INST]",
    # "[INST] Describe an eye-catching outfit someone might wear to a fancy party. [/INST]",
    # "[INST] What is a good name for a Samoyed dog? [/INST]",
    # "[INST] Please tell me how the Male Gaze affects depiction of women in video games. [/INST]",
    # "[INST] Please summarize the history of Pride Month. [/INST]",
    # "[INST] Please write me python code to compute the prime factors of a number. [/INST]",
    # "[INST] Please write me a Harry Potter fanfic about Roon Wazlib. [/INST]"
]

prompts = ["[INST ] <<SYS>>\n" + DEFAULT_SYSTEM_PROMPT + "\n<</SYS>>\n\n" + p[6:] for p in prompts]

# print("warming up...")
# tokenizer.pad_token = tokenizer.bos_token
# tokenizer.padding_side = 'left';
# inputs = tokenizer(prompts, return_tensors='pt', padding=True)
# # outputs = model.generate(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda(), max_new_tokens=32, min_new_tokens=32, do_sample=True, temperature=0.7, top_p=0.7, top_k=16, return_d>
# for i in range(5):
# 	outputs = model.generate(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda(), max_new_tokens=32, min_new_tokens=32, do_sample=True, temperature=0.7, top_p=0.7, top_k=16, return_dict_in_generate=True)
# print("warmed up!")

# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#     with record_function("model_inference"):

start = time.time()

# prompt = """It is a truth universally acknowledged that"""
tokenizer.pad_token = tokenizer.bos_token
tokenizer.padding_side = 'left'
inputs = tokenizer(prompts, return_tensors='pt', padding=True)
# outputs = model.generate(input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda(), max_new_tokens=32, min_new_tokens=32, do_sample=True, temperature=0.7, top_p=0.7, top_k=16, return_dict_in_generate=True)
outputs = model.generate(input_ids=inputs['input_ids'].cuda(),
                         attention_mask=inputs['attention_mask'].cuda(),
                         max_new_tokens=512,
                         min_new_tokens=512,
                         do_sample=True,
                         temperature=0.7,
                         top_p=0.7,
                         top_k=16,
                         return_dict_in_generate=True)
end = time.time()
for ip in range(8):
    token = outputs.sequences[ip, :]
    output_str = tokenizer.decode(token)
    print("\n")
    print(output_str)
    print("\n")

print(f"elapsed: {end - start}")

# print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=40))

# print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
