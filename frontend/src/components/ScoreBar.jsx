import React from 'react';
import { motion } from 'framer-motion';
import { getRiskColorRaw, normalizeScore } from '../utils/risk';

export default function ScoreBar({ name, score }) {
  const percentage = normalizeScore(score);
  const color = getRiskColorRaw(percentage);
  const clamped = Math.min(100, Math.max(0, percentage));

  return (
    <div className="mb-3">
      <div className="flex justify-between items-center mb-1">
        <span className="text-xs font-semibold text-[#5B4E75]">{name}</span>
        <span className="text-xs font-mono font-bold text-[#1E1238]">
          {percentage.toFixed(1)}%
        </span>
      </div>
      <div className="h-2 w-full rounded-full overflow-hidden bg-purple-100 border border-purple-200/50">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${clamped}%` }}
          transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
          className="h-full rounded-full"
          style={{ background: color }}
        />
      </div>
    </div>
  );
}
