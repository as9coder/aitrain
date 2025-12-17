// Prompt variations to ensure unique outputs
export const styleVariations = [
  'modern and minimalist',
  'colorful and vibrant',
  'dark mode themed',
  'glassmorphic design',
  'brutalist style',
  'neumorphic design',
  'retro/vintage style',
  'futuristic/cyberpunk',
  'professional corporate',
  'playful and fun',
];

export const featureAddons = [
  'with animations and transitions',
  'with mobile-first responsive design',
  'with accessibility features',
  'with real-time updates',
  'with search functionality',
  'with filtering and sorting',
  'with dark/light mode toggle',
  'with keyboard shortcuts',
  'with drag and drop',
  'with infinite scroll',
];

export const technicalPreferences = [
  'using React hooks',
  'using TypeScript strict mode',
  'with custom hooks',
  'with context API',
  'with local storage',
  'with form validation',
  'with error boundaries',
  'with lazy loading',
  'with memoization',
  'with performance optimization',
];

export function getRandomItem<T>(array: T[]): T {
  return array[Math.floor(Math.random() * array.length)];
}

export function shouldAddVariation(probability: number = 0.5): boolean {
  return Math.random() < probability;
}

export function generateUniquePrompt(baseType: string): string {
  let prompt = `Create ${baseType}`;
  
  // 70% chance to add style variation
  if (shouldAddVariation(0.7)) {
    prompt += ` with ${getRandomItem(styleVariations)}`;
  }
  
  // 60% chance to add feature addon
  if (shouldAddVariation(0.6)) {
    prompt += ` ${getRandomItem(featureAddons)}`;
  }
  
  // 50% chance to add technical preference
  if (shouldAddVariation(0.5)) {
    prompt += ` ${getRandomItem(technicalPreferences)}`;
  }
  
  return prompt;
}
