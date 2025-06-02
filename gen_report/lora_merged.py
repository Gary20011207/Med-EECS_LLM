import os
import torch
import shutil
from peft import PeftModel, PeftConfig, set_peft_model_state_dict
from safetensors.torch import load_file, save_file

# Set the path for each fold's LoRA weights and the output path for the merged result
fold_dirs = ["./qwen25vl_lora_fold1", "./qwen25vl_lora_fold2", "./qwen25vl_lora_fold3"]
merged_dir = "./qwen25vl_lora_merged"

# Load a single LoRA weight
def load_lora_weights(fold_path: str) -> dict:
    path_bin = os.path.join(fold_path, "adapter_model.bin")
    path_safe = os.path.join(fold_path, "adapter_model.safetensors")

    if os.path.exists(path_safe):
        return load_file(path_safe, device="cpu")
    elif os.path.exists(path_bin):
        return torch.load(path_bin, map_location="cpu")
    else:
        raise FileNotFoundError(f"No adapter weight found in {fold_path}")

# Collect state_dicts from all folds
state_dicts = [load_lora_weights(p) for p in fold_dirs]

# Average tensor weights
merged_state_dict = {}
for key in state_dicts[0]:
    if isinstance(state_dicts[0][key], torch.Tensor):
        merged_state_dict[key] = sum(d[key] for d in state_dicts) / len(state_dicts)
    else:
        merged_state_dict[key] = state_dicts[0][key]

# Create the output directory for the merged result
os.makedirs(merged_dir, exist_ok=True)

# Save as .safetensors format
save_file(merged_state_dict, os.path.join(merged_dir, "adapter_model.safetensors"))
print("Merged weights saved as .safetensors")

# Automatically copy the LoRA configuration file (adapter_config.json)
config_file = "adapter_config.json"
src_config = os.path.join(fold_dirs[0], config_file)
dst_config = os.path.join(merged_dir, config_file)
if os.path.exists(src_config):
    shutil.copy(src_config, dst_config)
    print("adapter_config.json copied successfully")
else:
    print("adapter_config.json not found. Inference will fail without it.")

# Optional: Copy tokenizer-related files (if available)
tokenizer_files = ["tokenizer_config.json", "tokenizer.model", "special_tokens_map.json"]
for fname in tokenizer_files:
    src = os.path.join(fold_dirs[0], fname)
    dst = os.path.join(merged_dir, fname)
    if os.path.exists(src):
        shutil.copy(src, dst)

print(f"LoRA weights merged and saved at: {merged_dir}")