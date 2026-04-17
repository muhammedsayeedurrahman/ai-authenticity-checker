import React from 'react';
import { motion } from 'framer-motion';

export default function ScoreBar({ name, score }) {
  // Assume score is between 0 and 1. If it's larger (e.g. 100), handle it.
  const percentage = score <= 1 ? score * 100 : score;
  
  const getRiskColor = (pct) => {
    if (pct > 70) return '#EC4899'; // High
    if (pct > 40) return '#F59E0B'; // Medium
    return '#10B981'; // Low
  };

  const color = getRiskColor(percentage);
  
  return (
    <div className="mb-3">
      <div className="flex justify-between items-end mb-1.5">
        <span className="text-xs font-medium text-text-secondary uppercase tracking-wider">{name}</span>
        <span className="text-xs font-bold" style={{ color }}>{percentage.toFixed(1)}%</span>
      </div>
      <div className="h-2 w-full bg-border-subtle rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${Math.min(100, Math.max(0, percentage))}%` }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="h-full rounded-full"
          style={{ 
            backgroundColor: color,
            boxShadow: `0 0 10px ${color}80`
          }}
        />
      </div>
    </div>
  );
}
