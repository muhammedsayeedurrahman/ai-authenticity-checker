import React, { useEffect, useState } from 'react';

export default function RiskGauge({ percentage, label = "AI Risk", size = 200 }) {
  const [offset, setOffset] = useState(0);
  
  const radius = size * 0.4;
  const circumference = 2 * Math.PI * radius;
  
  // Calculate risk level color
  const getRiskColor = (pct) => {
    if (pct > 70) return '#EC4899'; // High - Pink
    if (pct > 40) return '#F59E0B'; // Medium - Amber
    return '#10B981'; // Low - Green
  };

  const color = getRiskColor(percentage);
  
  useEffect(() => {
    const progressOffset = ((100 - percentage) / 100) * circumference;
    // Small timeout to trigger animation on mount
    const timer = setTimeout(() => setOffset(progressOffset), 100);
    return () => clearTimeout(timer);
  }, [percentage, circumference]);

  return (
    <div className="flex flex-col items-center justify-center relative p-6">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        {/* Background Circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth="12"
        />
        {/* Progress Circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="12"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset === 0 ? circumference : offset}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          style={{ transition: 'stroke-dashoffset 1s ease-out', filter: `drop-shadow(0 0 8px ${color}66)` }}
        />
      </svg>
      <div className="absolute flex flex-col items-center justify-center inset-0 pointer-events-none">
        <span className="text-4xl font-black font-sans tracking-tighter" style={{ color }}>
          {percentage.toFixed(1)}%
        </span>
        <span className="text-xs font-semibold text-text-secondary mt-1 uppercase tracking-wider">
          {label}
        </span>
      </div>
    </div>
  );
}
