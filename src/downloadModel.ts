import { spawn } from 'child_process';
import { existsSync, mkdirSync } from 'fs';
import { join } from 'path';

const MODEL_NAME = 'Qwen/Qwen2.5-Coder-1.5B-Instruct';
const MODEL_DIR = './models/qwen2.5-coder-1.5b-instruct';

async function downloadModel() {
  console.log('='.repeat(60));
  console.log('Downloading Qwen2.5-Coder-1.5B-Instruct');
  console.log('='.repeat(60));
  console.log(`Model: ${MODEL_NAME}`);
  console.log(`Destination: ${MODEL_DIR}`);
  console.log('='.repeat(60));
  
  // Create models directory if it doesn't exist
  if (!existsSync('./models')) {
    mkdirSync('./models', { recursive: true });
  }
  
  if (existsSync(MODEL_DIR)) {
    console.log('\n⚠️  Model directory already exists. Checking for updates...\n');
  }
  
  console.log('📥 Downloading model files (this may take a while, ~3GB)...\n');
  
  // Use huggingface-cli to download the model
  const args = [
    '-m', 'pip', 'install', '--quiet', 'huggingface_hub[cli]'
  ];
  
  console.log('Installing huggingface-cli...');
  const installProcess = spawn('python', args, { 
    stdio: 'inherit',
    shell: true 
  });
  
  installProcess.on('close', (code) => {
    if (code !== 0) {
      console.error('❌ Failed to install huggingface-cli');
      return;
    }
    
    console.log('\n✓ huggingface-cli installed\n');
    console.log('Downloading model...\n');
    
    // Download the model
    const downloadArgs = [
      '-m', 'huggingface_hub.commands.huggingface_cli',
      'download',
      MODEL_NAME,
      '--local-dir', MODEL_DIR,
      '--local-dir-use-symlinks', 'False'
    ];
    
    const downloadProcess = spawn('python', downloadArgs, { 
      stdio: 'inherit',
      shell: true 
    });
    
    downloadProcess.on('close', (downloadCode) => {
      if (downloadCode !== 0) {
        console.error('\n❌ Download failed');
        console.log('\nAlternative: Download manually from https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct');
        return;
      }
      
      console.log('\n' + '='.repeat(60));
      console.log('✓ Model downloaded successfully!');
      console.log('='.repeat(60));
      console.log(`Location: ${MODEL_DIR}`);
      console.log('\nModel includes:');
      console.log('  - config.json');
      console.log('  - model.safetensors');
      console.log('  - tokenizer files');
      console.log('  - generation config');
      console.log('\nReady for fine-tuning!');
      console.log('='.repeat(60));
    });
  });
}

downloadModel().catch(console.error);
