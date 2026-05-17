import React from 'react';

export default function NeuralBackground() {
  return (
    <div
      className="fixed inset-0 pointer-events-none z-0"
      style={{ overflow: 'hidden' }}
    >
      {/* Orb 1 — top-left blue wash */}
      <div
        style={{
          position: 'absolute',
          top: '-10%',
          left: '-5%',
          width: '60%',
          height: '60%',
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(59,130,246,0.12) 0%, transparent 70%)',
          filter: 'blur(80px)',
          animation: 'orbFloat1 20s ease-in-out infinite',
        }}
      />

      {/* Orb 2 — bottom-right cyan wash */}
      <div
        style={{
          position: 'absolute',
          bottom: '-15%',
          right: '-10%',
          width: '50%',
          height: '55%',
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(6,182,212,0.08) 0%, transparent 70%)',
          filter: 'blur(80px)',
          animation: 'orbFloat2 25s ease-in-out infinite',
        }}
      />

      {/* Orb 3 — center blue accent */}
      <div
        style={{
          position: 'absolute',
          top: '30%',
          left: '40%',
          width: '40%',
          height: '40%',
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(59,130,246,0.06) 0%, transparent 70%)',
          filter: 'blur(60px)',
          animation: 'orbFloat3 18s ease-in-out infinite',
        }}
      />

      {/* Scanline overlay */}
      <div
        style={{
          position: 'absolute',
          inset: 0,
          backgroundImage:
            'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(255,255,255,0.015) 2px, rgba(255,255,255,0.015) 4px)',
          opacity: 0.4,
        }}
      />

      <style>{`
        @keyframes orbFloat1 {
          0%, 100% { transform: translate(0, 0) scale(1); }
          33% { transform: translate(3%, 5%) scale(1.05); }
          66% { transform: translate(-2%, -3%) scale(0.97); }
        }
        @keyframes orbFloat2 {
          0%, 100% { transform: translate(0, 0) scale(1); }
          40% { transform: translate(-4%, -3%) scale(1.03); }
          70% { transform: translate(2%, 4%) scale(0.98); }
        }
        @keyframes orbFloat3 {
          0%, 100% { transform: translate(0, 0) scale(1); }
          50% { transform: translate(5%, -4%) scale(1.06); }
        }
      `}</style>
    </div>
  );
}
