/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Neon Obsidian palette — Cybersecurity Blue
        bg: {
          void:     '#080A0F',
          base:     '#0C0F16',
          card:     '#141820',
          elevated: '#1C2130',
          inset:    '#0A0C12',
        },
        border: {
          dim:  'rgba(255,255,255,0.07)',
          mid:  'rgba(255,255,255,0.12)',
        },
        accent: {
          DEFAULT: '#3B82F6',
          dim:     'rgba(59,130,246,0.12)',
          glow:    'rgba(59,130,246,0.18)',
          2:       '#06B6D4',
        },
        text: {
          1: '#EDF0F7',
          2: '#8B95A5',
          3: '#4A5264',
        },
        risk: {
          clear:    '#34D399',
          caution:  '#FBBF24',
          critical: '#FB7185',
          clearDim:    'rgba(52,211,153,0.10)',
          cautionDim:  'rgba(251,191,36,0.10)',
          criticalDim: 'rgba(251,113,133,0.10)',
        },
        // Compatibility aliases
        ink: {
          primary: '#EDF0F7',
          body:    '#8B95A5',
          muted:   '#4A5264',
        },
      },
      boxShadow: {
        'card':       '0 2px 8px rgba(0,0,0,0.5), 0 0 0 1px rgba(255,255,255,0.04), inset 0 1px 0 rgba(255,255,255,0.03)',
        'card-hover': '0 8px 32px rgba(0,0,0,0.6), 0 0 0 1px rgba(59,130,246,0.20), 0 0 20px rgba(59,130,246,0.15)',
        'modal':      '0 24px 64px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.05)',
        'glow-blue':   '0 0 20px rgba(59,130,246,0.20)',
        'glow-cyan':   '0 0 20px rgba(6,182,212,0.10)',
        'glow-green':  '0 0 12px rgba(52,211,153,0.15)',
        'glow-red':    '0 0 12px rgba(251,113,133,0.15)',
        'glow-amber':  '0 0 12px rgba(251,191,36,0.15)',
      },
      fontFamily: {
        sans:    ['Inter', 'system-ui', 'sans-serif'],
        mono:    ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
        display: ['"Space Grotesk"', 'Inter', 'sans-serif'],
      },
      borderRadius: {
        DEFAULT: '10px',
      },
      animation: {
        'fade-in':    'fadeIn 0.35s ease-out',
        'slide-up':   'slideUp 0.4s cubic-bezier(0.22, 1, 0.36, 1)',
        'shimmer':    'shimmer 2s linear infinite',
        'pulse-glow': 'pulseGlow 3s ease-in-out infinite',
        'float':      'float 6s ease-in-out infinite',
      },
      keyframes: {
        fadeIn:    { '0%': { opacity: 0 }, '100%': { opacity: 1 } },
        slideUp:   { '0%': { opacity: 0, transform: 'translateY(12px)' }, '100%': { opacity: 1, transform: 'translateY(0)' } },
        shimmer:   { '0%': { backgroundPosition: '-200% 0' }, '100%': { backgroundPosition: '200% 0' } },
        pulseGlow: {
          '0%, 100%': { boxShadow: '0 0 8px rgba(59,130,246,0.20)' },
          '50%':      { boxShadow: '0 0 24px rgba(59,130,246,0.40)' },
        },
        float: {
          '0%, 100%': { transform: 'translateY(0)' },
          '50%':      { transform: 'translateY(-6px)' },
        },
      },
    },
  },
  plugins: [],
}
