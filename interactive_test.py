"""
Interactive test for the fine-tuned model
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_PATH = "./models/qwen2.5-coder-1.5b-instruct"
FINE_TUNED_PATH = "./fine-tuned-model"

def load_model():
    print("Loading model... (this may take a moment)")
    
    tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_PATH, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    base_model.resize_token_embeddings(len(tokenizer))
    model = PeftModel.from_pretrained(base_model, FINE_TUNED_PATH)
    model.eval()
    
    return model, tokenizer

def clean_output(text):
    """Remove artifact tokens that aren't part of our format"""
    import re
    
    # Remove common artifact patterns (non-ASCII between valid tags)
    # This preserves code/content but removes random unicode chars
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # If line contains TOOL_CALL, remove non-ASCII chars outside JSON
        if '<TOOL_CALL>' in line:
            # Extract the JSON part and clean the rest
            start = line.find('<TOOL_CALL>')
            end = line.find('</TOOL_CALL>')
            if start != -1 and end != -1:
                before = line[:start]
                json_part = line[start:end+12]  # Include closing tag
                # Remove non-ASCII from before section
                before = re.sub(r'[^\x00-\x7F]+', '', before)
                cleaned_lines.append(before + json_part)
                continue
        
        # For other lines, keep as is if mostly ASCII or if it's code
        cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def generate(model, tokenizer, prompt, max_tokens=2048):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Get token IDs for stopping
    end_token = tokenizer.encode("<|end|>", add_special_tokens=False)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,  # Lower temperature for more focused output
            top_p=0.85,
            top_k=50,  # Add top-k sampling to reduce random tokens
            do_sample=True,
            repetition_penalty=1.1,  # Penalize repetition
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=end_token[0] if end_token else tokenizer.eos_token_id,
            no_repeat_ngram_size=3,  # Prevent repeating 3-grams
        )
    
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return clean_output(decoded)

def main():
    print("="*60)
    print("Fine-tuned Model Interactive Test")
    print("="*60)
    
    model, tokenizer = load_model()
    print("\n✓ Model loaded successfully!\n")
    
    test_prompts = [
        "Create a simple todo list app",
        "Create a weather dashboard with temperature display",
        "Create a login form with email and password",
    ]
    
    print("Choose a test prompt:")
    for i, prompt in enumerate(test_prompts, 1):
        print(f"  {i}. {prompt}")
    print(f"  {len(test_prompts) + 1}. Custom prompt")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == str(len(test_prompts) + 1):
        user_prompt = input("Enter your prompt: ").strip()
    elif choice.isdigit() and 1 <= int(choice) <= len(test_prompts):
        user_prompt = test_prompts[int(choice) - 1]
    else:
        print("Invalid choice, using default")
        user_prompt = test_prompts[0]
    
    # Format prompt with our special tokens
    formatted_prompt = f"<|start|><|user|>{user_prompt}<|assistant|><|tool_start|>"
    
    print("\n" + "="*60)
    print(f"Generating response for: {user_prompt}")
    print("="*60)
    print("\nGenerating...\n")
    
    response = generate(model, tokenizer, formatted_prompt, max_tokens=1024)
    
    # Extract just the generated part
    generated = response[len(formatted_prompt):]
    
    print("RESPONSE:")
    print("-"*60)
    print(generated)
    print("-"*60)

if __name__ == "__main__":
    main()
