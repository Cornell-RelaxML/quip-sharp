import time
import torch
from transformers import LlamaTokenizer, LlamaForCausalLM

model = LlamaForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', torch_dtype="auto", low_cpu_mem_usage=True)
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

torch.set_grad_enabled(False)

model.half()
model.cuda()

model = torch.compile(model)

print("generating some text...")

# B_INST, E_INST = "[INST]", "[/INST]"
# B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
# DEFAULT_SYSTEM_PROMPT = """\
# You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

# If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

prompts = [
    "[INST] What fun things can I do during my conference trip to Honolulu? [/INST] ",
    "[INST] Write me the introductory paragraph for a Harry Potter fanfic. [/INST] ",
    "[INST] Tell me your Grandmother's recipe for Spanikopita. [/INST] ",
    "[INST] What is the best chess opening for a beginner? [/INST] ",
    "[INST] What is the difference between a 401-k and a Roth IRA? [/INST] ",
    "[INST] Tell me the fairy tail of the mouse, the frog, and the sausage. [/INST] ",
    "[INST] In Les Miserables, why was Jean Valjean arrested? [/INST] ",
    "[INST] Please list all the US presidents in order. [/INST] "
]

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
                         return_dict_in_generate=True)
for ip in range(8):
    token = outputs.sequences[ip, :]
    output_str = tokenizer.decode(token)
    print("\n")
    print(output_str)
    print("\n")

end = time.time()

print(f"elapsed: {end - start}")
