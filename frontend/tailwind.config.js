/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Single source of truth: CSS custom properties in index.css :root
        bg: {
          void:     'var(--bg-void)',
          base:     'var(--bg-base)',
          card:     'var(--bg-card)',
          elevated: 'var(--bg-elevated)',
          inset:    'var(--bg-inset)',
        },
        border: {
          dim:  'var(--border-dim)',
          mid:  'var(--border-mid)',
          glow: 'var(--border-glow)',
        },
        accent: {
          DEFAULT: 'var(--accent)',
          dim:     'var(--accent-dim)',
          glow:    'var(--accent-glow)',
          2:       'var(--accent-2)',
        },
        text: {
          1: 'var(--text-1)',
          2: 'var(--text-2)',
          3: 'var(--text-3)',
        },
        risk: {
          clear:       'var(--risk-clear)',
          caution:     'var(--risk-caution)',
          critical:    'var(--risk-critical)',
          clearDim:    'rgba(34,197,94,0.10)',
          cautionDim:  'rgba(250,204,21,0.10)',
          criticalDim: 'rgba(251,113,133,0.10)',
        },
      },
      boxShadow: {
        'card':       '0 4px 24px rgba(0,0,0,0.45), 0 0 0 1px rgba(255,255,255,0.04), inset 0 1px 0 rgba(255,255,255,0.03)',
        'card-hover': '0 12px 40px rgba(0,0,0,0.55), 0 0 0 1px rgba(59,130,246,0.22), 0 0 24px rgba(59,130,246,0.16)',
        'modal':      '0 24px 64px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.05)',
        'glow-blue':   '0 0 20px rgba(59,130,246,0.20)',
        'glow-cyan':   '0 0 20px rgba(56,189,248,0.10)',
        'glow-green':  '0 0 12px rgba(34,197,94,0.15)',
        'glow-red':    '0 0 12px rgba(251,113,133,0.15)',
        'glow-amber':  '0 0 12px rgba(250,204,21,0.15)',
      },
      fontFamily: {
        sans:    ['Inter', 'system-ui', 'sans-serif'],
        mono:    ['"JetBrains Mono"', '"Fira Code"', 'monospace'],
        display: ['"Space Grotesk"', 'Inter', 'sans-serif'],
      },
      borderRadius: {
        DEFAULT: '12px',
        // Distinct key (not `lg`, which 17+ files already use for small
        // icon boxes/toasts at Tailwind's default 8px) — the generous
        // 18-22px "premium" radius is applied only via .card/.card-elevated
        // in index.css, which reference var(--radius-lg) directly.
      },
      animation: {
        'fade-in':  'fadeIn 0.35s ease-out',
        'slide-up': 'slideUp 0.4s cubic-bezier(0.22, 1, 0.36, 1)',
        'shimmer':  'shimmer 2s linear infinite',
      },
      keyframes: {
        fadeIn:  { '0%': { opacity: 0 }, '100%': { opacity: 1 } },
        slideUp: { '0%': { opacity: 0, transform: 'translateY(12px)' }, '100%': { opacity: 1, transform: 'translateY(0)' } },
        shimmer: { '0%': { backgroundPosition: '-200% 0' }, '100%': { backgroundPosition: '200% 0' } },
      },
    },
  },
  plugins: [],
}
