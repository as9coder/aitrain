#!/bin/bash
# Quick setup script for H100 cloud training

echo "=================================="
echo "AI Training Setup - H100 Cloud"
echo "=================================="

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -q -r requirements.txt

# Download base model
echo "📥 Downloading Qwen2.5-Coder-1.5B-Instruct..."
python -m huggingface_hub.commands.huggingface_cli download \
    Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --local-dir ./models/qwen2.5-coder-1.5b-instruct \
    --local-dir-use-symlinks False

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Upload your training_data.jsonl file to this directory"
echo "2. Run: python train.py"
echo ""
echo "Expected training time on H100: 1.5-2 hours"
echo "=================================="
