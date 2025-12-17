export const websiteTypes = [
  // Landing Pages
  'a modern SaaS landing page with hero section, features, and pricing',
  'a minimalist portfolio landing page for a photographer',
  'a startup landing page with animated elements and call-to-action',
  'a product launch landing page with countdown timer',
  'an app landing page with download buttons and screenshots',
  'a conference event landing page with speakers and schedule',
  'a restaurant landing page with menu and reservation form',
  'a gym/fitness landing page with class schedule and trainers',
  'a real estate landing page with property listings',
  'a crypto/blockchain project landing page',
  
  // Dashboards
  'an analytics dashboard with charts and metrics',
  'a project management dashboard with task lists and progress',
  'an e-commerce admin dashboard with sales data',
  'a social media analytics dashboard',
  'a fitness tracking dashboard with workout stats',
  'a financial portfolio dashboard with stock charts',
  'a CRM dashboard with customer data and pipeline',
  'an IoT device monitoring dashboard',
  'a content management dashboard',
  'a team collaboration dashboard',
  
  // Forms & Data Entry
  'a multi-step form for user onboarding',
  'a job application form with file upload',
  'a survey form with various input types',
  'a contact form with validation',
  'a booking/reservation form with date picker',
  'a payment checkout form with credit card input',
  'a registration form with email verification',
  'a quiz/assessment form with scoring',
  'a feedback form with rating system',
  'an order form with product selection',
  
  // E-commerce
  'a product listing page with filters and sorting',
  'a product detail page with image gallery and reviews',
  'a shopping cart page with quantity controls',
  'a checkout page with shipping options',
  'a user profile page with order history',
  'a wishlist page with save for later',
  'a category page with breadcrumb navigation',
  'a search results page for products',
  'a deals/promotions page with countdown timers',
  'a subscription box selection page',
  
  // Social & Community
  'a social media feed with posts and comments',
  'a user profile page with bio and activity',
  'a messaging interface with chat threads',
  'a forum thread page with replies',
  'a community dashboard with members and activity',
  'a notification center with activity feed',
  'a friends/connections page with search',
  'a groups/communities listing page',
  'an events calendar with RSVP',
  'a photo sharing gallery page',
  
  // Content & Media
  'a blog homepage with article cards',
  'a blog post page with comments',
  'a video player page with recommendations',
  'a podcast player page with episodes list',
  'an image gallery with lightbox',
  'a news portal homepage',
  'a documentation page with sidebar navigation',
  'a recipe page with ingredients and steps',
  'a course listing page for online learning',
  'a music player interface with playlist',
  
  // Productivity & Tools
  'a todo list application with categories',
  'a notes taking app with rich text editor',
  'a calendar app with event management',
  'a kanban board for task management',
  'a timer/pomodoro app with statistics',
  'a habit tracker with progress visualization',
  'a budget/expense tracker app',
  'a bookmark manager with tags',
  'a password manager interface',
  'a code snippet manager',
  
  // Interactive & Creative
  'a drawing/whiteboard app with tools',
  'a color palette generator',
  'a meme generator with text overlay',
  'a resume builder with templates',
  'a invoice generator with PDF export',
  'a QR code generator',
  'a markdown editor with live preview',
  'a CSS gradient generator',
  'a lorem ipsum generator with options',
  'a emoji picker component',
  
  // Gaming & Entertainment
  'a tic-tac-toe game',
  'a memory card matching game',
  'a quiz game with score tracking',
  'a typing speed test game',
  'a trivia game with categories',
  'a word guessing game',
  'a dice rolling simulator',
  'a slot machine game',
  'a sudoku game interface',
  'a chess board interface',
  
  // Professional & Business
  'a pricing comparison table',
  'a team members page with profiles',
  'a testimonials/reviews page',
  'a FAQ page with accordion',
  'a careers/jobs listing page',
  'a case studies showcase page',
  'a services page with feature comparison',
  'a about us page with company timeline',
  'a press/media kit page',
  'a partner/sponsor showcase page',
  
  // Specialized
  'a weather app with forecast',
  'a currency converter',
  'a BMI calculator with health tips',
  'a loan/mortgage calculator',
  'a unit converter (length, weight, temp)',
  'a timezone converter',
  'a age calculator',
  'a calorie counter',
  'a tip calculator for restaurants',
  'a compound interest calculator',
];

export function getRandomWebsiteType(): string {
  return websiteTypes[Math.floor(Math.random() * websiteTypes.length)];
}

export function getComplexity(type: string): 'simple' | 'medium' | 'complex' {
  const simpleKeywords = ['calculator', 'converter', 'generator', 'timer', 'counter'];
  const complexKeywords = ['dashboard', 'social', 'e-commerce', 'management', 'analytics'];
  
  const typeLower = type.toLowerCase();
  
  if (complexKeywords.some(keyword => typeLower.includes(keyword))) {
    return 'complex';
  }
  
  if (simpleKeywords.some(keyword => typeLower.includes(keyword))) {
    return 'simple';
  }
  
  return 'medium';
}
