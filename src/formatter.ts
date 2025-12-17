import type { TrainingExample, FormattedTrainingData, ToolCall } from './types.js';

// Special tokens for training
export const TOKENS = {
  START: '<|start|>',
  END: '<|end|>',
  USER: '<|user|>',
  ASSISTANT: '<|assistant|>',
  TOOL_START: '<|tool_start|>',
  TOOL_END: '<|tool_end|>',
  CODE_START: '<|code_start|>',
  CODE_END: '<|code_end|>',
};

export function formatForTraining(example: TrainingExample): FormattedTrainingData {
  let formatted = `${TOKENS.START}${TOKENS.USER}${example.prompt}${TOKENS.ASSISTANT}`;
  
  // Add tool calls section
  if (example.tool_calls.length > 0) {
    formatted += `${TOKENS.TOOL_START}\n`;
    
    for (const toolCall of example.tool_calls) {
      formatted += `<TOOL_CALL>${JSON.stringify(toolCall)}</TOOL_CALL>\n`;
    }
    
    formatted += `${TOKENS.TOOL_END}`;
  }
  
  // Add code section
  formatted += `${TOKENS.CODE_START}\n${example.code}\n${TOKENS.CODE_END}${TOKENS.END}`;
  
  return { text: formatted };
}

export function parseGeminiResponse(response: string): { tool_calls: ToolCall[], code: string } {
  const tool_calls: ToolCall[] = [];
  let code = '';
  
  // Extract tool calls
  const toolCallsMatch = response.match(/=== TOOL CALLS START ===([\s\S]*?)=== TOOL CALLS END ===/);
  if (toolCallsMatch) {
    const toolCallsSection = toolCallsMatch[1];
    const toolCallMatches = toolCallsSection.matchAll(/<TOOL_CALL>(.*?)<\/TOOL_CALL>/g);
    
    for (const match of toolCallMatches) {
      try {
        const toolCall = JSON.parse(match[1]);
        tool_calls.push(toolCall);
      } catch (error) {
        console.error('Failed to parse tool call:', match[1]);
      }
    }
  }
  
  // Extract code
  const codeMatch = response.match(/=== CODE START ===([\s\S]*?)=== CODE END ===/);
  if (codeMatch) {
    code = codeMatch[1].trim();
  }
  
  return { tool_calls, code };
}
