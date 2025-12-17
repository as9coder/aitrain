export interface ToolCall {
  type: 'create_file' | 'read_file' | 'edit_file' | 'delete_file' | 'create_directory' | 'list_directory';
  path: string;
  content?: string;
  oldContent?: string;
  newContent?: string;
}

export interface TrainingExample {
  prompt: string;
  tool_calls: ToolCall[];
  code: string;
  metadata: {
    website_type: string;
    complexity: 'simple' | 'medium' | 'complex';
    components_count: number;
    generated_at: string;
  };
}

export interface FormattedTrainingData {
  text: string; // The complete formatted text with special tokens
}
