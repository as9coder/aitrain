import { readFileSync } from 'fs';
import type { FormattedTrainingData } from './types.js';

const OUTPUT_FILE = 'training_data.jsonl';

function analyzeTrainingData() {
  console.log('Analyzing training data...\n');
  
  const lines = readFileSync(OUTPUT_FILE, 'utf-8').trim().split('\n');
  
  console.log(`Total examples: ${lines.length}`);
  
  let totalChars = 0;
  let totalToolCalls = 0;
  const complexities: Record<string, number> = {};
  
  for (const line of lines) {
    const data: FormattedTrainingData = JSON.parse(line);
    totalChars += data.text.length;
    
    const toolCallMatches = data.text.match(/<TOOL_CALL>/g);
    if (toolCallMatches) {
      totalToolCalls += toolCallMatches.length;
    }
  }
  
  console.log(`Total characters: ${totalChars.toLocaleString()}`);
  console.log(`Average chars per example: ${Math.round(totalChars / lines.length).toLocaleString()}`);
  console.log(`Total tool calls: ${totalToolCalls}`);
  console.log(`Average tool calls per example: ${(totalToolCalls / lines.length).toFixed(2)}`);
  
  // Show a sample
  console.log('\n--- Sample Training Example ---');
  const sample: FormattedTrainingData = JSON.parse(lines[0]);
  console.log(sample.text.substring(0, 500) + '...\n');
}

analyzeTrainingData();
