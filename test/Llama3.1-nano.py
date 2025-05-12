import torch
import transformers

model_id = "nvidia/Llama-3.1-Nemotron-Nano-8B-v1"
model_kwargs = {"torch_dtype": torch.bfloat16, "device_map": "auto"}
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id

pipeline = transformers.pipeline(
   "text-generation",   
   model=model_id,
   tokenizer=tokenizer,
   max_new_tokens=32768,
   temperature=0.6,
   top_p=0.95,
   **model_kwargs
)

# Thinking can be "on" or "off"
thinking = "on"

print(pipeline([{"role": "system", "content": f"detailed thinking {thinking}"}, {"role": "user", "content": "Solve x*(sin(x)+2)=0"}]))
