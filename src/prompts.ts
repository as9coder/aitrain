export function buildPromptForGemini(userRequest: string): string {
  return `You are an expert web developer creating React + TypeScript website examples for training an AI model.

The AI model you're generating data for needs to learn:
1. How to understand user requests for websites
2. How to make appropriate tool calls to set up files and directories
3. How to write clean React + TypeScript code

USER REQUEST: "${userRequest}"

Generate a COMPLETE example that includes:

1. TOOL CALLS SECTION - List all necessary tool calls to set up the project:
   - create_directory: Create necessary folders (src, src/components, src/styles, etc.)
   - create_file: Create all necessary files (App.tsx, components, styles, etc.)
   - Format: <TOOL_CALL>{"type": "create_directory", "path": "src"}</TOOL_CALL>

2. CODE SECTION - Provide the complete implementation:
   - Main App.tsx component
   - All necessary sub-components
   - CSS/styling (use Tailwind CSS classes or inline styles)
   - TypeScript types/interfaces
   - Make it functional and production-ready

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

=== TOOL CALLS START ===
<TOOL_CALL>{"type": "create_directory", "path": "src"}</TOOL_CALL>
<TOOL_CALL>{"type": "create_directory", "path": "src/components"}</TOOL_CALL>
<TOOL_CALL>{"type": "create_file", "path": "src/App.tsx", "content": ""}</TOOL_CALL>
... (all other tool calls)
=== TOOL CALLS END ===

=== CODE START ===
// src/App.tsx
import React from 'react';
... (complete code implementation)
=== CODE END ===

Requirements:
- Use modern React hooks (useState, useEffect, etc.)
- Include proper TypeScript types
- Make it visually appealing
- Add responsive design
- Include interactive elements where appropriate
- Keep code clean and well-commented

Generate the complete response now:`;
}

export function buildSystemPrompt(): string {
  return `You are a code generation AI that outputs structured tool calls followed by code implementations. Always follow the exact format specified in the prompt.`;
}
