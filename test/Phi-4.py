import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
torch.random.manual_seed(0)

model_id = "microsoft/Phi-4-mini-reasoning"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

messages = [{
    "role": "user",
    "content": "How to solve 3*x^2+4*x+5=1?"
}]   
inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
)

outputs = model.generate(
    **inputs.to(model.device),
    max_new_tokens=32768,
    temperature=0.8,
    top_p=0.95,
    do_sample=True,
)
outputs = tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[-1]:])

print(outputs[0])
