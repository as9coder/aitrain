import { writeFileSync, appendFileSync, existsSync, readFileSync } from 'fs';
import { generateWithGemini } from './gemini.js';
import { buildPromptForGemini } from './prompts.js';
import { formatForTraining, parseGeminiResponse } from './formatter.js';
import { getRandomWebsiteType, getComplexity } from './websiteTypes.js';
import { generateUniquePrompt } from './promptVariations.js';
import type { TrainingExample } from './types.js';

const OUTPUT_FILE = 'training_data.jsonl';
const TARGET_EXAMPLES = 8000; // Target 8k examples for robust training
const BATCH_SIZE = 20; // Send 20 requests in parallel

// Track generated prompts to ensure uniqueness
const generatedPrompts = new Set<string>();

function generateUniquePromptWithRetry(): string {
  const websiteType = getRandomWebsiteType();
  let userPrompt = generateUniquePrompt(websiteType);
  
  // Ensure uniqueness - regenerate if duplicate
  let attempts = 0;
  while (generatedPrompts.has(userPrompt) && attempts < 20) {
    const newType = getRandomWebsiteType();
    userPrompt = generateUniquePrompt(newType);
    attempts++;
  }
  
  generatedPrompts.add(userPrompt);
  return userPrompt;
}

async function generateSingleExample(index: number, userPrompt: string): Promise<boolean> {
  try {
    const geminiPrompt = buildPromptForGemini(userPrompt);
    const response = await generateWithGemini(geminiPrompt);
    
    const { tool_calls, code } = parseGeminiResponse(response);
    
    if (tool_calls.length === 0 || !code) {
      console.warn(`⚠️  [${index}] Invalid response format, skipping...`);
      return false;
    }
    
    const websiteType = userPrompt.replace('Create ', '').split(' with ')[0];
    const example: TrainingExample = {
      prompt: userPrompt,
      tool_calls,
      code,
      metadata: {
        website_type: websiteType,
        complexity: getComplexity(websiteType),
        components_count: (code.match(/function|const.*?=.*?=>/g) || []).length,
        generated_at: new Date().toISOString(),
      }
    };
    
    const formattedData = formatForTraining(example);
    
    // Append to JSONL file
    appendFileSync(OUTPUT_FILE, JSON.stringify(formattedData) + '\n');
    
    console.log(`✓ [${index}] Generated successfully (${tool_calls.length} tool calls, ${code.length} chars)`);
    return true;
    
  } catch (error) {
    console.error(`✗ [${index}] Error:`, error instanceof Error ? error.message : error);
    return false;
  }
}

async function generateBatch(startIndex: number, batchSize: number): Promise<number> {
  // Generate unique prompts for this batch
  const prompts = Array.from({ length: batchSize }, () => generateUniquePromptWithRetry());
  
  console.log(`\n🚀 Starting batch ${Math.floor(startIndex / BATCH_SIZE) + 1}: Generating ${batchSize} examples in parallel...`);
  prompts.forEach((prompt, i) => {
    console.log(`   [${startIndex + i}] ${prompt}`);
  });
  
  // Send all requests in parallel
  const promises = prompts.map((prompt, i) => 
    generateSingleExample(startIndex + i, prompt)
  );
  
  const results = await Promise.all(promises);
  const successCount = results.filter(r => r).length;
  
  console.log(`\n✓ Batch complete: ${successCount}/${batchSize} successful`);
  return successCount;
}

async function main() {
  console.log('='.repeat(60));
  console.log('Website Training Data Generator');
  console.log('='.repeat(60));
  console.log(`Target: ${TARGET_EXAMPLES} examples`);
  console.log(`Output: ${OUTPUT_FILE}`);
  console.log('='.repeat(60));
  
  // Count existing examples in file
  let existingCount = 0;
  if (existsSync(OUTPUT_FILE)) {
    const content = readFileSync(OUTPUT_FILE, 'utf-8').trim();
    if (content) {
      existingCount = content.split('\n').filter(line => line.trim()).length;
      console.log(`\n📂 Found ${existingCount} existing examples, continuing from there...`);
    }
  } else {
    // Create file if it doesn't exist
    writeFileSync(OUTPUT_FILE, '');
  }
  
  const remainingToGenerate = TARGET_EXAMPLES - existingCount;
  if (remainingToGenerate <= 0) {
    console.log(`\n✓ Already have ${existingCount} examples (target: ${TARGET_EXAMPLES}). Nothing to generate.`);
    return;
  }
  
  console.log(`Generating ${remainingToGenerate} more examples...\n`);
  
  let totalSuccess = 0;
  const totalBatches = Math.ceil(remainingToGenerate / BATCH_SIZE);
  const startTime = Date.now();
  
  // Process in batches
  for (let batchNum = 0; batchNum < totalBatches; batchNum++) {
    const startIndex = existingCount + batchNum * BATCH_SIZE + 1;
    const remainingExamples = remainingToGenerate - (batchNum * BATCH_SIZE);
    const currentBatchSize = Math.min(BATCH_SIZE, remainingExamples);
    
    const batchSuccess = await generateBatch(startIndex, currentBatchSize);
    totalSuccess += batchSuccess;
    
    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
    const currentTotal = existingCount + totalSuccess;
    const progress = ((currentTotal / TARGET_EXAMPLES) * 100).toFixed(1);
    console.log(`\n📊 Progress: ${currentTotal}/${TARGET_EXAMPLES} (${progress}%) | Time: ${elapsed}s | Batch ${batchNum + 1}/${totalBatches}`);
  }
  
  const totalTime = ((Date.now() - startTime) / 1000).toFixed(1);
  const finalTotal = existingCount + totalSuccess;
  console.log('\n' + '='.repeat(60));
  console.log(`✓ Generation complete!`);
  console.log(`  Total examples: ${finalTotal}/${TARGET_EXAMPLES}`);
  console.log(`  Generated this session: ${totalSuccess}`);
  console.log(`  Total time: ${totalTime}s`);
  console.log(`  Average: ${(totalSuccess / parseFloat(totalTime)).toFixed(2)} examples/sec`);
  console.log(`  Output file: ${OUTPUT_FILE}`);
  console.log('='.repeat(60));
}

main().catch(console.error);
