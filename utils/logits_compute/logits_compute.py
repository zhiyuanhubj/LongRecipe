from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json

prompt = ['Hello, world, greate world']
# prompt = ['Hugging Face is creating great transformers models!']
sampling_params = SamplingParams(prompt_logprobs=1)
model_path = 'meta-llama/Meta-Llama-3-8B'

tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = LLM(model=model_path,trust_remote_code=True,gpu_memory_utilization=0.9)

tokenized_input = tokenizer(prompt[0])
tokens = tokenizer.convert_ids_to_tokens(tokenized_input['input_ids'])
ids = tokenized_input['input_ids']
ids.pop(0)
tokens.pop(0)

try:
    outputs = llm.generate(prompt,sampling_params)
except:
    pass

with open('prompt_logits.json','r') as f:
    logits = json.load(f)
    
def save_to_json(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f)
        
save_to_json('prompt_tokens_logits.json', dict(zip(tokens, logits[0])))  ## chagne to your customized path
save_to_json('prompt_ids_logits.json', dict(zip(ids, logits[0]))) ## chagne to your customized path
