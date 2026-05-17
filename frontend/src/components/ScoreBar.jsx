import React from 'react';
import { motion } from 'framer-motion';

export default function ScoreBar({ name, score }) {
  const percentage = score <= 1 ? score * 100 : score;

  const getColor = (pct) => {
    if (pct > 70) return '#FB7185';
    if (pct > 40) return '#FBBF24';
    return '#34D399';
  };

  const getGlow = (pct) => {
    if (pct > 70) return '0 0 8px rgba(251,113,133,0.3)';
    if (pct > 40) return '0 0 8px rgba(251,191,36,0.3)';
    return '0 0 8px rgba(52,211,153,0.3)';
  };

  const color = getColor(percentage);
  const glow = getGlow(percentage);
  const clamped = Math.min(100, Math.max(0, percentage));

  return (
    <div className="mb-2.5">
      <div className="flex justify-between items-center mb-1.5">
        <span
          className="text-[11px]"
          style={{ color: 'var(--text-2, #8B95A5)' }}
        >
          {name}
        </span>
        <span
          className="text-[11px] font-mono"
          style={{ color: 'var(--text-1, #EDF0F7)' }}
        >
          {percentage.toFixed(1)}%
        </span>
      </div>
      {/* Track */}
      <div
        className="h-1 w-full rounded-full overflow-hidden"
        style={{ background: 'rgba(255,255,255,0.06)' }}
      >
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${clamped}%` }}
          transition={{ duration: 0.9, ease: [0.22, 1, 0.36, 1] }}
          className="h-full rounded-full"
          style={{ background: color, boxShadow: glow }}
        />
      </div>
    </div>
  );
}
