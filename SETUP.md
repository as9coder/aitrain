# AI Training Data Generator & Fine-tuning Pipeline

Generate training data using Gemini API and fine-tune Qwen2.5-Coder-1.5B for React/TypeScript code generation with tool calling.

## 🚀 Quick Start (H100 Cloud)

### 1. Clone Repository
```bash
git clone https://github.com/as9coder/aitrain.git
cd aitrain
```

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Training Data
You'll need the `training_data.jsonl` file (not in repo due to size). Download it separately or generate new data.

### 4. Download Base Model
```bash
python -m huggingface_hub.commands.huggingface_cli download Qwen/Qwen2.5-Coder-1.5B-Instruct --local-dir ./models/qwen2.5-coder-1.5b-instruct --local-dir-use-symlinks False
```

### 5. Start Training
```bash
python train.py
```

## ⚡ H100 Optimization

For H100 GPUs, the training will automatically use optimal settings. Expected training time: **1.5-2 hours** for 5600 examples × 15 epochs.

## 📁 Project Structure

```
aitrain/
├── src/                    # TypeScript source for data generation
│   ├── generate.ts         # Main generation script
│   ├── gemini.ts          # Gemini API integration
│   ├── prompts.ts         # Prompt templates
│   ├── formatter.ts       # JSONL formatting
│   └── websiteTypes.ts    # 100+ website types
├── train.py               # Training script (RTX 4060 / H100)
├── test_model.py          # Test fine-tuned model
├── interactive_test.py    # Interactive testing
├── requirements.txt       # Python dependencies
├── package.json          # Node.js dependencies
└── training_data.jsonl   # Training data (not in repo)
```

## 🎯 Training Configuration

Current setup (optimized for quality):
- **Examples:** 5600
- **Epochs:** 15
- **Max Sequence Length:** 4096 tokens
- **Batch Size:** 1 (RTX 4060) or 8 (H100)
- **LoRA Rank:** 16 (RTX 4060) or 32 (H100)
- **Total Training Steps:** ~84,000

## 📊 Model Output Format

The model learns to generate:
1. **Tool calls** for file operations (create_directory, create_file, etc.)
2. **Complete React/TypeScript code** for websites
3. **Proper formatting** with special tokens

Example output:
```
<|start|><|user|>Create a todo app<|assistant|><|tool_start|>
<TOOL_CALL>{"type":"create_directory","path":"src"}</TOOL_CALL>
<TOOL_CALL>{"type":"create_file","path":"src/App.tsx","content":""}</TOOL_CALL>
<|tool_end|><|code_start|>
// React code here
<|code_end|><|end|>
```

## 🔧 Training on Different Hardware

### RTX 4060 (8GB VRAM)
Uses current `train.py` settings:
- 8-bit quantization
- Batch size 1
- ~20-25 hours training time

### H100 (80GB VRAM)
Automatic optimizations:
- Full precision (BF16)
- Batch size 8
- Higher LoRA rank
- ~1.5-2 hours training time

## 📝 Generating More Training Data

If you need to generate training data from scratch:

1. **Set up Gemini API:**
   ```bash
   cp .env.example .env
   # Edit .env and add your GEMINI_API_KEY
   ```

2. **Install Node.js dependencies:**
   ```bash
   npm install
   ```

3. **Generate data:**
   ```bash
   npm run generate
   ```

## 🧪 Testing the Model

After training completes:

```bash
# Simple test
python test_model.py

# Interactive test
python interactive_test.py
```

## 💾 Checkpoints

- Saves every 250 steps
- Keeps last 5 checkpoints
- Located in `./checkpoints/`
- Final model saved to `./fine-tuned-model/`

## 📈 Monitoring Training

Training displays:
- Real-time progress bar with ETA
- Loss and learning rate
- Checkpoint notifications
- Truncation warnings (if any examples are too long)

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size` to 1
- Reduce `max_seq_length` to 2048
- Enable 8-bit quantization

### Model Quality Issues
- Increase epochs (15-20)
- Ensure training data isn't truncated
- Check that all examples have proper format

### Slow Training
- Use BF16 on newer GPUs
- Increase batch size if VRAM allows
- Use gradient accumulation

## 📄 License

MIT

## 🤝 Contributing

Feel free to open issues or submit PRs!
