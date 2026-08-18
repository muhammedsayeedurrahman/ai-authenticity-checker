import React from 'react';

// Fixed positions/timings (not randomized per render) so the effect is
// stable and predictable rather than reshuffling on every mount.
const PARTICLES = [
  { top: '12%', left: '18%', size: 3, color: 'rgba(59,130,246,0.5)',  glow: 'rgba(59,130,246,0.6)',  duration: 22, delay: 0, driftX: 14,  driftY: -18 },
  { top: '68%', left: '8%',  size: 2, color: 'rgba(56,189,248,0.45)',  glow: 'rgba(56,189,248,0.5)',   duration: 26, delay: 2, driftX: -10, driftY: 12 },
  { top: '24%', left: '82%', size: 3, color: 'rgba(59,130,246,0.4)',  glow: 'rgba(59,130,246,0.5)',  duration: 30, delay: 4, driftX: -16, driftY: 10 },
  { top: '80%', left: '72%', size: 2, color: 'rgba(56,189,248,0.5)',   glow: 'rgba(56,189,248,0.55)',  duration: 20, delay: 1, driftX: 12,  driftY: -14 },
  { top: '45%', left: '92%', size: 2, color: 'rgba(59,130,246,0.35)', glow: 'rgba(59,130,246,0.45)', duration: 24, delay: 6, driftX: -8,  driftY: -16 },
  { top: '55%', left: '38%', size: 2, color: 'rgba(56,189,248,0.35)',  glow: 'rgba(56,189,248,0.4)',   duration: 28, delay: 3, driftX: 10,  driftY: 14 },
  { top: '8%',  left: '55%', size: 2, color: 'rgba(59,130,246,0.4)',  glow: 'rgba(59,130,246,0.5)',  duration: 25, delay: 5, driftX: -12, driftY: 16 },
  { top: '92%', left: '30%', size: 3, color: 'rgba(56,189,248,0.4)',   glow: 'rgba(56,189,248,0.5)',   duration: 32, delay: 7, driftX: 8,   driftY: -10 },
];

/**
 * Layered ambient background: static radial glow + a very subtle drifting
 * grid + a handful of slow-floating glow particles.
 *
 * Every animated property is background-position / transform / opacity
 * only — no animated blur, no layout-triggering properties — keeping this
 * cheap on top of a fixed full-viewport layer. Fully covered by the global
 * prefers-reduced-motion rule in index.css.
 */
export default function NeuralBackground() {
  return (
    <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden" aria-hidden="true">
      {/* Static radial glow */}
      <div
        className="absolute inset-0"
        style={{
          background: `
            radial-gradient(ellipse 60% 50% at 15% 10%, rgba(59,130,246,0.07) 0%, transparent 60%),
            radial-gradient(ellipse 50% 40% at 85% 90%, rgba(56,189,248,0.04) 0%, transparent 60%)
          `,
        }}
      />

      {/* Subtle drifting grid */}
      <div className="bg-grid" />

      {/* Slow-floating glow particles */}
      {PARTICLES.map((p, i) => (
        <span
          key={i}
          className="bg-particle"
          style={{
            top: p.top,
            left: p.left,
            width: p.size,
            height: p.size,
            background: p.color,
            boxShadow: `0 0 ${p.size * 4}px ${p.glow}`,
            animationDuration: `${p.duration}s`,
            animationDelay: `${p.delay}s`,
            '--particle-drift-x': `${p.driftX}px`,
            '--particle-drift-y': `${p.driftY}px`,
          }}
        />
      ))}

      {/* Static grain — removes the "flat digital surface" look. Not
          animated (single paint, no ongoing cost). */}
      <div
        className="absolute inset-0"
        style={{
          opacity: 0.02,
          backgroundImage: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")",
          backgroundRepeat: 'repeat',
        }}
      />

      {/* Edge vignette */}
      <div
        className="absolute inset-0"
        style={{
          background: 'radial-gradient(ellipse 80% 80% at 50% 50%, transparent 55%, rgba(0,0,0,0.35) 100%)',
        }}
      />

      {/* Mouse spotlight — fed by --spotlight-x/--spotlight-y, set by the
          shared rAF-throttled pointermove listener in Layout.jsx. */}
      <div
        className="absolute inset-0"
        style={{
          background: 'radial-gradient(300px circle at var(--spotlight-x, 50%) var(--spotlight-y, 50%), rgba(59,130,246,0.03), transparent 70%)',
        }}
      />
    </div>
  );
}
