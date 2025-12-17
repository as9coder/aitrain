# H100 optimized settings (80GB VRAM)
TRAINING_CONFIG_H100 = {
    "batch_size": 8,  # Much larger batch size
    "gradient_accumulation_steps": 2,  # Effective batch size = 16
    "learning_rate": 3e-4,  # Higher LR for larger batches
    "num_epochs": 15,
    "max_seq_length": 4096,
    "warmup_steps": 150,
    "save_steps": 250,
    "logging_steps": 10,
    "fp16": False,  # H100 supports bf16 better
    "bf16": True,  # Use BF16 instead of FP16
    "save_total_limit": 5,
    "lr_scheduler": "cosine",
    "weight_decay": 0.01,
}

# LoRA config - can increase rank on H100
LORA_CONFIG_H100 = {
    "r": 32,  # Higher rank (more capacity)
    "lora_alpha": 64,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # More modules
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}
