from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

model_path = "./"
save_path = "./qwen_with_lidar"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True)

new_tokens = {"additional_special_tokens": ["<lidar>"]}
num_added_toks = tokenizer.add_special_tokens(new_tokens)

print(f"Added {num_added_toks} tokens.")
print(f"New <lidar> token ID: {tokenizer.convert_tokens_to_ids('<lidar>')}")

model.resize_token_embeddings(len(tokenizer))

tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)
print(f"Saved updated model and tokenizer to {save_path}")