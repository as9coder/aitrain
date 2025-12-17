# Fine-tuning Qwen2.5-Coder-1.5B-Instruct

## Training Setup

Optimized for **RTX 4060 (8GB VRAM)**

### Features
- ✅ 8-bit quantization (saves VRAM)
- ✅ LoRA fine-tuning (efficient training)
- ✅ Custom special tokens for tool calling
- ✅ Gradient checkpointing (memory optimization)
- ✅ Mixed precision training (FP16)

### Memory Usage
- **Model (8-bit):** ~2GB
- **Training overhead:** ~4-5GB
- **Total:** ~6-7GB (fits comfortably in 8GB)

## Installation

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Training

Once you have completed generating 1500 training examples, start training:

```bash
python train.py
```

### Training Time (RTX 4060)
- **1500 examples** × **3 epochs** = 4500 steps
- **Estimated time:** ~6-8 hours
- **Effective batch size:** 8 (1 × 8 gradient accumulation)

## Configuration

Edit `train.py` to adjust:

### For more VRAM (12GB+ GPU):
```python
TRAINING_CONFIG = {
    "batch_size": 2,  # Increase batch size
    "gradient_accumulation_steps": 4,
    "max_seq_length": 4096,  # Longer sequences
}
```

### For less VRAM (6GB GPU):
```python
TRAINING_CONFIG = {
    "batch_size": 1,
    "gradient_accumulation_steps": 16,  # More accumulation
    "max_seq_length": 1024,  # Shorter sequences
}
```

### LoRA settings:
```python
LORA_CONFIG = {
    "r": 16,  # Higher = more capacity, more VRAM
    "lora_alpha": 32,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}
```

## Testing the Model

After training completes, test it:

```bash
python test_model.py
```

This will generate a response for a sample prompt using your fine-tuned model.

## Output Structure

```
aitrain/
├── fine-tuned-model/       # Final trained model
│   ├── adapter_model.bin   # LoRA weights
│   ├── adapter_config.json
│   └── tokenizer files
├── checkpoints/            # Training checkpoints
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── ...
└── logs/                   # Training logs
```

## Special Tokens

The model is trained to recognize these tokens:
- `<|start|>` / `<|end|>` - Example boundaries
- `<|user|>` / `<|assistant|>` - Conversation flow
- `<|tool_start|>` / `<|tool_end|>` - Tool calling section
- `<|code_start|>` / `<|code_end|>` - Code section
- `<TOOL_CALL>` / `</TOOL_CALL>` - Individual tool calls

## Inference Example

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("./fine-tuned-model")
model = AutoModelForCausalLM.from_pretrained(
    "./fine-tuned-model",
    torch_dtype=torch.float16,
    device_map="auto"
)

prompt = "<|start|><|user|>Create a todo list app<|assistant|><|tool_start|>"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

outputs = model.generate(**inputs, max_new_tokens=1024)
print(tokenizer.decode(outputs[0]))
```

## Troubleshooting

### Out of Memory Error
- Reduce `batch_size` to 1
- Increase `gradient_accumulation_steps`
- Reduce `max_seq_length`
- Use 4-bit quantization instead of 8-bit

### Slow Training
- Enable gradient checkpointing (already enabled)
- Use smaller LoRA rank (r=8 instead of 16)
- Reduce sequence length

### Model not converging
- Increase learning rate to 3e-4
- Train for more epochs
- Increase LoRA rank to 32
