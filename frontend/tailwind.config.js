/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: {
          DEFAULT: '#09090B',
          surface: '#18181B',
          elevated: '#27272A',
          card: 'rgba(255, 255, 255, 0.04)',
          'card-hover': 'rgba(255, 255, 255, 0.07)',
        },
        border: {
          subtle: 'rgba(255, 255, 255, 0.06)',
          hover: 'rgba(99, 102, 241, 0.25)',
        },
        accent: {
          DEFAULT: '#6366F1',
          glow: 'rgba(99, 102, 241, 0.15)',
          success: '#22C55E',
          warning: '#EAB308',
          danger: '#EF4444',
        },
        text: {
          primary: '#FAFAFA',
          secondary: '#A1A1AA',
          muted: '#71717A',
        },
        risk: {
          low: '#22C55E',
          medium: '#EAB308',
          high: '#EF4444',
        },
      },
      boxShadow: {
        'glow-accent': '0 0 20px rgba(99, 102, 241, 0.15)',
        'glow-success': '0 0 20px rgba(34, 197, 94, 0.15)',
        'glow-danger': '0 0 20px rgba(239, 68, 68, 0.15)',
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'system-ui', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
