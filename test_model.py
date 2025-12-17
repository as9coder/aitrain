"""
Test the fine-tuned model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_PATH = "./models/qwen2.5-coder-1.5b-instruct"
FINE_TUNED_PATH = "./fine-tuned-model"

def test_model():
    print("Loading base model and tokenizer...")
    
    # Load tokenizer with special tokens
    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_PATH, trust_remote_code=True)
    
    # Load base model
    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Resize embeddings for new tokens
    base_model.resize_token_embeddings(len(tokenizer))
    
    # Load LoRA adapter
    print("Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(base_model, FINE_TUNED_PATH)
    model.eval()
    
    # Test prompt
    test_prompt = "<|start|><|user|>Create a simple counter app with increment and decrement buttons<|assistant|><|tool_start|>"
    
    print("\n" + "="*60)
    print("Test Prompt:")
    print(test_prompt)
    print("="*60)
    print("\nGenerating response...\n")
    
    inputs = tokenizer(test_prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=False)
    print(response)
    print("\n" + "="*60)

if __name__ == "__main__":
    test_model()
