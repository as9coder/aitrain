# Website Training Data Generator

Generate 1k-1.5k unique website examples using Gemini API for fine-tuning a code generation model.

## Features

- 🤖 Uses Gemini API to generate diverse React + TypeScript websites
- 🛠️ Includes tool calls (create/delete files, read, edit, etc.)
- 📝 Outputs structured JSONL format with special tokens for fine-tuning
- 🎨 100+ different website types (dashboards, landing pages, forms, games, etc.)
- 🔄 Built-in rate limiting and error handling

## Setup

1. Install dependencies:
```bash
npm install
```

2. Create `.env` file with your Gemini API key:
```bash
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

Get your API key from: https://makersuite.google.com/app/apikey

## Usage

### Generate Training Data

Run the generation script to create 1.5k examples:

```bash
npm run generate
```

This will:
- Generate 1500 unique website examples
- Save them to `training_data.jsonl`
- Include tool calls and complete code for each example
- Show progress updates every 10 examples

### Analyze Generated Data

Check statistics about your generated training data:

```bash
npm run dev src/analyze.ts
```

## Data Format

Each line in `training_data.jsonl` contains a structured training example:

```json
{
  "text": "<|start|><|user|>Create a todo list application<|assistant|><|tool_start|>\n<TOOL_CALL>{...}</TOOL_CALL>\n<|tool_end|><|code_start|>\n// code here\n<|code_end|><|end|>"
}
```

### Special Tokens

- `<|start|>` / `<|end|>` - Example boundaries
- `<|user|>` - User prompt section
- `<|assistant|>` - AI response section
- `<|tool_start|>` / `<|tool_end|>` - Tool calls section
- `<|code_start|>` / `<|code_end|>` - Code section
- `<TOOL_CALL>` - Individual tool call wrapper

### Tool Call Types

- `create_directory` - Create folders
- `create_file` - Create files with content
- `read_file` - Read file contents
- `edit_file` - Modify existing files
- `delete_file` - Remove files
- `list_directory` - List directory contents

## Project Structure

```
aitrain/
├── src/
│   ├── generate.ts       # Main generation script
│   ├── analyze.ts        # Data analysis script
│   ├── gemini.ts         # Gemini API integration
│   ├── prompts.ts        # Prompt templates
│   ├── formatter.ts      # JSONL formatting with tokens
│   ├── websiteTypes.ts   # 100+ website types
│   └── types.ts          # TypeScript interfaces
├── training_data.jsonl   # Generated output (gitignored)
└── package.json
```

## Customization

### Add More Website Types

Edit `src/websiteTypes.ts` to add more website examples:

```typescript
export const websiteTypes = [
  'your custom website description',
  // ... more types
];
```

### Adjust Generation Parameters

In `src/generate.ts`:
- `TARGET_EXAMPLES`: Number of examples to generate (default: 1500)
- `DELAY_MS`: Delay between API calls (default: 1000ms)

In `src/gemini.ts`:
- `temperature`: Controls randomness (0.0-2.0)
- `maxOutputTokens`: Maximum response length

## Fine-tuning

Once you have `training_data.jsonl`, you can use it to fine-tune your 1.5B parameter model. The format includes:

1. **User prompts** - What the user wants to build
2. **Tool calls** - File operations needed
3. **Code** - Complete React + TypeScript implementation
4. **Special tokens** - For the model to learn structure

Your model will learn to:
- Understand natural language requests for websites
- Generate appropriate tool calls for file setup
- Write clean, functional React + TypeScript code
- Follow the structured output format

## Tips

- Start with a small batch (50-100 examples) to test your setup
- Monitor API usage and rate limits
- The generation process takes ~25 minutes for 1500 examples (with 1s delay)
- Review a few examples to ensure quality before generating the full dataset

## License

MIT
