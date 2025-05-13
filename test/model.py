from core.model_manager import ModelManager

manager = ModelManager()
model, tokenizer, model_max_length = manager.initialize()

model = manager.get_model()

text = "This is a test sentence!!1"

token_num = manager.count_tokens(text)
print(f"Token number: {token_num}")