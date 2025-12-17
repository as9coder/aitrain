"""
Fine-tune Qwen2.5-Coder-1.5B-Instruct on RTX 4060 (8GB VRAM)
Uses LoRA for memory efficiency
"""

import os
import json
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import bitsandbytes as bnb
from tqdm import tqdm
import time

# Configuration
MODEL_PATH = "./models/qwen2.5-coder-1.5b-instruct"
TRAINING_DATA = "./training_data.jsonl"
OUTPUT_DIR = "./fine-tuned-model"
CHECKPOINT_DIR = "./checkpoints"

# Special tokens from our training data format
SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "<|start|>",
        "<|end|>",
        "<|user|>",
        "<|assistant|>",
        "<|tool_start|>",
        "<|tool_end|>",
        "<|code_start|>",
        "<|code_end|>",
        "<TOOL_CALL>",
        "</TOOL_CALL>"
    ]
}

# Auto-detect GPU and set optimal config
def get_training_config():
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_name = torch.cuda.get_device_name(0).lower()
        
        # H200 / H100 / A100 (80GB+)
        if gpu_memory > 70 or 'h100' in gpu_name or 'h200' in gpu_name or 'a100' in gpu_name:
            # H200 has 141GB - maximize it!
            batch_size = 64 if gpu_memory > 130 else 32  # H200 gets 64, H100 gets 32
            return {
                "batch_size": batch_size,  # MASSIVE batch to use all VRAM
                "gradient_accumulation_steps": 1,  # No accumulation needed
                "learning_rate": 5e-4,  # Higher LR for large batches
                "num_epochs": 8,  # Fewer epochs with huge batches
                "max_seq_length": 8192,  # Double the sequence length
                "warmup_steps": 200,
                "save_steps": 100,  # Save more often (faster epochs)
                "logging_steps": 5,
                "fp16": False,
                "bf16": True,  # BF16 for H100/H200
                "save_total_limit": 5,
                "lr_scheduler": "cosine",
                "weight_decay": 0.01,
                "use_8bit": False,  # No quantization
                "lora_rank": 64,  # Much higher rank
                "dataloader_num_workers": 4,  # More workers
            }
        # RTX 4090 / A6000 (24GB)
        elif gpu_memory > 20:
            return {
                "batch_size": 4,
                "gradient_accumulation_steps": 2,
                "learning_rate": 2e-4,
                "num_epochs": 12,
                "max_seq_length": 4096,
                "warmup_steps": 150,
                "save_steps": 250,
                "logging_steps": 10,
                "fp16": True,
                "bf16": False,
                "save_total_limit": 5,
                "lr_scheduler": "cosine",
                "weight_decay": 0.01,
                "use_8bit": True,
            }
    
    # RTX 4060 / Small GPU (8GB)
    return {
        "batch_size": 1,
        "gradient_accumulation_steps": 8,
        "learning_rate": 2e-4,
        "num_epochs": 15,
        "max_seq_length": 4096,
        "warmup_steps": 150,
        "save_steps": 250,
        "logging_steps": 10,
        "fp16": True,
        "bf16": False,
        "save_total_limit": 5,
        "lr_scheduler": "cosine",
        "weight_decay": 0.01,
        "use_8bit": True,
    }

TRAINING_CONFIG = get_training_config()

# LoRA configuration - will be updated based on GPU
def get_lora_config():
    rank = TRAINING_CONFIG.get("lora_rank", 16)
    
    # More target modules for better quality
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
        "gate_proj", "up_proj", "down_proj"  # MLP layers
    ]
    
    return {
        "r": rank,
        "lora_alpha": rank * 2,  # Alpha = 2x rank
        "target_modules": target_modules,
        "lora_dropout": 0.05,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    }

LORA_CONFIG = get_lora_config()

def load_and_prepare_data():
    """Load JSONL training data with content validation"""
    print("📂 Loading training data...")
    
    data = []
    truncated_count = 0
    total_chars = 0
    
    with open(TRAINING_DATA, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                text = item["text"]
                data.append({"text": text})
                
                total_chars += len(text)
                
                # Check if example has all expected parts
                has_tool_calls = "<|tool_start|>" in text and "<|tool_end|>" in text
                has_code = "<|code_start|>" in text and "<|code_end|>" in text
                has_end = "<|end|>" in text
                
                if not (has_tool_calls and has_code and has_end):
                    truncated_count += 1
    
    avg_length = total_chars // len(data) if data else 0
    print(f"✓ Loaded {len(data)} training examples")
    print(f"  Average length: {avg_length:,} characters")
    if truncated_count > 0:
        print(f"  ⚠️  {truncated_count} examples may be truncated or incomplete")
    
    return Dataset.from_list(data)

def setup_model_and_tokenizer():
    """Load model with 8-bit quantization and prepare for LoRA"""
    print("🤖 Loading model and tokenizer...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        use_fast=False
    )
    
    # Add special tokens
    print("🔧 Adding special tokens...")
    num_added = tokenizer.add_special_tokens(SPECIAL_TOKENS)
    print(f"✓ Added {num_added} special tokens")
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate quantization
    use_8bit = TRAINING_CONFIG.get("use_8bit", True)
    dtype = torch.bfloat16 if TRAINING_CONFIG.get("bf16", False) else torch.float16
    
    print(f"💾 Loading model {'with 8-bit quantization' if use_8bit else 'in full precision'}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        load_in_8bit=use_8bit,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=dtype
    )
    
    # Resize token embeddings for new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    # Prepare model for k-bit training (only if using quantization)
    if use_8bit:
        model = prepare_model_for_kbit_training(model)
    else:
        # For full precision training, enable gradient checkpointing
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    
    # Apply LoRA with dynamic config
    print("🔗 Applying LoRA configuration...")
    LORA_CONFIG = get_lora_config()  # Refresh config with current TRAINING_CONFIG
    print(f"   LoRA rank: {LORA_CONFIG['r']}, Target modules: {len(LORA_CONFIG['target_modules'])}")
    lora_config = LoraConfig(**LORA_CONFIG)
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer

def tokenize_dataset(dataset, tokenizer):
    """Tokenize the dataset with smart truncation"""
    print("🔤 Tokenizing dataset...")
    
    truncated_examples = 0
    total_examples = len(dataset)
    
    def tokenize_function(examples):
        nonlocal truncated_examples
        
        # Tokenize with truncation
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=TRAINING_CONFIG["max_seq_length"],
            padding="max_length",
            return_tensors=None  # Don't convert to tensors yet
        )
        
        # Check for truncation
        for i, text in enumerate(examples["text"]):
            original_tokens = tokenizer.encode(text, add_special_tokens=False)
            if len(original_tokens) > TRAINING_CONFIG["max_seq_length"]:
                truncated_examples += 1
        
        return result
    
    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    print(f"✓ Tokenization complete")
    if truncated_examples > 0:
        pct = (truncated_examples / total_examples) * 100
        print(f"  ⚠️  {truncated_examples}/{total_examples} examples ({pct:.1f}%) were truncated")
        print(f"     Consider increasing max_seq_length if this is high")
    
    return tokenized

class ProgressCallback(TrainerCallback):
    """Custom callback for progress tracking with ETA"""
    
    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.start_time = None
        self.pbar = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        self.pbar = tqdm(total=self.total_steps, desc="Training", unit="step")
        
    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.pbar and logs:
            current_step = state.global_step
            self.pbar.n = current_step
            
            # Calculate ETA
            elapsed = time.time() - self.start_time
            steps_per_sec = current_step / elapsed if elapsed > 0 else 0
            remaining_steps = self.total_steps - current_step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            
            # Format ETA
            eta_hours = int(eta_seconds // 3600)
            eta_mins = int((eta_seconds % 3600) // 60)
            eta_secs = int(eta_seconds % 60)
            
            # Update progress bar
            postfix = {}
            if 'loss' in logs:
                postfix['loss'] = f"{logs['loss']:.4f}"
            if 'learning_rate' in logs:
                postfix['lr'] = f"{logs['learning_rate']:.2e}"
            postfix['ETA'] = f"{eta_hours:02d}:{eta_mins:02d}:{eta_secs:02d}"
            
            self.pbar.set_postfix(postfix)
            self.pbar.refresh()
            
    def on_save(self, args, state, control, **kwargs):
        if self.pbar:
            print(f"\n💾 Checkpoint saved at step {state.global_step}")
            
    def on_train_end(self, args, state, control, **kwargs):
        if self.pbar:
            self.pbar.close()

def train():
    """Main training function"""
    print("="*60)
    print("Fine-tuning Qwen2.5-Coder-1.5B-Instruct")
    print("Optimized for RTX 4060 (8GB VRAM)")
    print("="*60)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
        print(f"📊 Training Config: batch_size={TRAINING_CONFIG['batch_size']}, seq_len={TRAINING_CONFIG['max_seq_length']}, LoRA_rank={TRAINING_CONFIG.get('lora_rank', 16)}")
        print(f"⚡ Estimated VRAM usage: ~{gpu_memory * 0.85:.0f}GB (targeting 85% utilization)")
    else:
        print("⚠️  No GPU detected! Training will be very slow.")
    
    # Load data
    dataset = load_and_prepare_data()
    
    # Setup model and tokenizer
    model, tokenizer = setup_model_and_tokenizer()
    
    # Tokenize dataset
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Calculate total training steps
    total_steps = (len(tokenized_dataset) // TRAINING_CONFIG["batch_size"] // TRAINING_CONFIG["gradient_accumulation_steps"]) * TRAINING_CONFIG["num_epochs"]
    print(f"\n📊 Training Configuration:")
    print(f"   Examples: {len(tokenized_dataset)}")
    print(f"   Epochs: {TRAINING_CONFIG['num_epochs']}")
    print(f"   Total steps: {total_steps}")
    print(f"   Checkpoint every: {TRAINING_CONFIG['save_steps']} steps")
    print(f"   Max checkpoints kept: {TRAINING_CONFIG['save_total_limit']}")
    
    # Training arguments with optimizations
    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=TRAINING_CONFIG["num_epochs"],
        per_device_train_batch_size=TRAINING_CONFIG["batch_size"],
        gradient_accumulation_steps=TRAINING_CONFIG["gradient_accumulation_steps"],
        learning_rate=TRAINING_CONFIG["learning_rate"],
        fp16=TRAINING_CONFIG["fp16"],
        save_steps=TRAINING_CONFIG["save_steps"],
        logging_steps=TRAINING_CONFIG["logging_steps"],
        warmup_steps=TRAINING_CONFIG["warmup_steps"],
        save_total_limit=TRAINING_CONFIG["save_total_limit"],
        logging_dir="./logs",
        report_to="none",  # Disable wandb/tensorboard
        optim="paged_adamw_8bit",  # Memory efficient optimizer
        gradient_checkpointing=True,  # Save VRAM
        max_grad_norm=1.0,
        disable_tqdm=True,  # Disable default tqdm, we use custom callback
        lr_scheduler_type=TRAINING_CONFIG["lr_scheduler"],  # Cosine schedule
        weight_decay=TRAINING_CONFIG["weight_decay"],  # Regularization
        dataloader_num_workers=2,  # Parallel data loading
        group_by_length=False,  # Don't group - ensure all parts seen equally
        length_column_name=None,  # Process examples as-is
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize progress callback
    progress_callback = ProgressCallback(total_steps)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[progress_callback],
    )
    
    # Start training
    print("\n🚀 Starting training...\n")
    trainer.train()
    
    # Save final model
    print("\n💾 Saving final model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("\n" + "="*60)
    print("✓ Training complete!")
    print("="*60)
    print(f"Model saved to: {OUTPUT_DIR}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}")
    print("\nTo use the model:")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{OUTPUT_DIR}')")
    print(f"  tokenizer = AutoTokenizer.from_pretrained('{OUTPUT_DIR}')")
    print("="*60)

if __name__ == "__main__":
    train()
