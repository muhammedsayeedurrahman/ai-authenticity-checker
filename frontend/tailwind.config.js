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
          DEFAULT: '#0A0E1A',
          card: 'rgba(255, 255, 255, 0.04)',
          'card-hover': 'rgba(255, 255, 255, 0.07)',
        },
        border: {
          subtle: 'rgba(255, 255, 255, 0.08)',
          glow: 'rgba(0, 240, 255, 0.15)',
        },
        accent: {
          cyan: '#00F0FF',
          violet: '#A855F7',
          pink: '#EC4899',
          green: '#10B981',
          amber: '#F59E0B',
        },
        text: {
          primary: '#E2E8F0',
          secondary: '#94A3B8',
          muted: '#64748B',
        },
        risk: {
          low: '#10B981',
          medium: '#F59E0B',
          high: '#EC4899',
        }
      },
      boxShadow: {
        'glow-cyan': '0 0 20px rgba(0, 240, 255, 0.15)',
        'glow-violet': '0 0 20px rgba(168, 85, 247, 0.15)',
        'glow-pink': '0 0 20px rgba(236, 72, 153, 0.15)',
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'system-ui', 'sans-serif'],
      }
    },
  },
  plugins: [],
}
