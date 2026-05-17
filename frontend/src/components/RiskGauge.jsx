import React, { useEffect, useState } from 'react';

export default function RiskGauge({ percentage, label = 'AI Risk Score', size = 160 }) {
  const [animated, setAnimated] = useState(false);

  const radius = size * 0.38;
  const circumference = 2 * Math.PI * radius;
  const strokeWidth = 6;

  const getColor = (pct) => {
    if (pct > 70) return '#FB7185';
    if (pct > 40) return '#FBBF24';
    return '#34D399';
  };

  const getRiskLabel = (pct) => {
    if (pct > 70) return 'High Risk';
    if (pct > 40) return 'Medium Risk';
    return 'Low Risk';
  };

  const getGlowColor = (pct) => {
    if (pct > 70) return 'rgba(251,113,133,0.4)';
    if (pct > 40) return 'rgba(251,191,36,0.4)';
    return 'rgba(52,211,153,0.4)';
  };

  const color = getColor(percentage);
  const glowColor = getGlowColor(percentage);
  const offset = animated ? ((100 - percentage) / 100) * circumference : circumference;

  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), 120);
    return () => clearTimeout(t);
  }, [percentage]);

  return (
    <div className="flex flex-col items-center justify-center py-4">
      <div
        className="relative"
        style={{
          width: size,
          height: size,
          transform: animated ? 'scale(1)' : 'scale(0.9)',
          opacity: animated ? 1 : 0,
          transition: 'transform 0.6s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.4s ease',
        }}
      >
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          style={{ transform: 'rotate(-90deg)' }}
        >
          {/* Glow filter */}
          <defs>
            <filter id={`glow-${size}`} x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Track */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.04)"
            strokeWidth={strokeWidth}
          />
          {/* Progress with glow */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth={strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            filter={`url(#glow-${size})`}
            style={{
              transition: 'stroke-dashoffset 1.1s cubic-bezier(0.22, 1, 0.36, 1)',
            }}
          />
        </svg>

        {/* Center text */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className="font-display font-bold tracking-tight"
            style={{ fontSize: '1.5rem', lineHeight: 1, color: 'var(--text-1, #EDF0F7)' }}
          >
            {percentage.toFixed(1)}<span style={{ fontSize: '0.8rem' }}>%</span>
          </span>
          <span
            className="text-[9px] uppercase tracking-[0.12em] mt-1"
            style={{ color: 'var(--text-3, #4A5264)' }}
          >
            {label}
          </span>
        </div>
      </div>

      {/* Risk level pill with glow */}
      <div
        className="mt-2 px-3 py-1 rounded-full text-[10px] font-medium uppercase tracking-wider"
        style={{
          background: `${color}10`,
          color: color,
          border: `1px solid ${color}26`,
          boxShadow: `0 0 12px ${glowColor}`,
        }}
      >
        {getRiskLabel(percentage)}
      </div>
    </div>
  );
}
