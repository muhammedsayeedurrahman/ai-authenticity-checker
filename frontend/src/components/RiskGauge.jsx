import React, { useEffect, useState } from 'react';
import { getRiskColorRaw, getRiskLevel } from '../utils/risk';
import AnimatedNumber from './AnimatedNumber';

export default function RiskGauge({
  percentage = 0,
  label = 'AI Risk',
  sublabel = null,
  size = 170,
  strokeWidth = 10,
  showBadge = true,
}) {
  const [animated, setAnimated] = useState(false);

  const radius = (size - strokeWidth * 2) / 2;
  const circumference = 2 * Math.PI * radius;
  const color = getRiskColorRaw(percentage);
  const clamped = Math.min(100, Math.max(0, percentage));
  const offset = animated ? ((100 - clamped) / 100) * circumference : circumference;

  useEffect(() => {
    const t = setTimeout(() => setAnimated(true), 100);
    return () => clearTimeout(t);
  }, [percentage]);

  const uniqueId = `gauge-grad-${Math.random().toString(36).substr(2, 9)}`;

  return (
    <div className="flex flex-col items-center justify-center">
      {/* Radial SVG Dial */}
      <div
        className="relative flex items-center justify-center"
        style={{
          width: size,
          height: size,
          transform: animated ? 'scale(1)' : 'scale(0.96)',
          opacity: animated ? 1 : 0,
          transition: 'transform 0.5s cubic-bezier(0.22, 1, 0.36, 1), opacity 0.3s ease',
        }}
      >
        <svg
          width={size}
          height={size}
          viewBox={`0 0 ${size} ${size}`}
          className="overflow-visible"
          style={{ transform: 'rotate(-90deg)' }}
        >
          <defs>
            <filter id={`glow-${uniqueId}`} x="-20%" y="-20%" width="140%" height="140%">
              <feDropShadow dx="0" dy="0" stdDeviation="3" floodColor={color} floodOpacity="0.4" />
            </filter>
          </defs>

          {/* Background Track */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="rgba(139, 92, 246, 0.12)"
            strokeWidth={strokeWidth}
          />

          {/* Animated Active Arc */}
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
            filter={`url(#glow-${uniqueId})`}
            style={{
              transition: 'stroke-dashoffset 1.1s cubic-bezier(0.22, 1, 0.36, 1)',
            }}
          />
        </svg>

        {/* Center Content: Pure Numbers & Clean Short Tag */}
        <div className="absolute inset-0 flex flex-col items-center justify-center text-center p-2 pointer-events-none">
          <div className="flex items-baseline justify-center font-display font-black tracking-tight leading-none text-[#1E1238]">
            <span className="text-3xl sm:text-4xl">
              <AnimatedNumber value={percentage} decimals={1} />
            </span>
            <span className="text-base sm:text-lg text-purple-700 ml-0.5 font-bold">%</span>
          </div>

          {label && (
            <span className="text-[10px] font-bold uppercase tracking-widest text-[#8F81A8] mt-1 truncate max-w-[110px]">
              {label}
            </span>
          )}
        </div>
      </div>

      {/* Risk Badge Below Circle */}
      {showBadge && (
        <div
          className="mt-3 px-3.5 py-1 rounded-full text-xs font-bold uppercase tracking-wider flex items-center gap-1.5 shadow-sm"
          style={{
            background: `${color}14`,
            color: color,
            border: `1px solid ${color}35`,
          }}
        >
          <span
            className="w-2 h-2 rounded-full animate-pulse"
            style={{ backgroundColor: color }}
          />
          {getRiskLevel(percentage)}
        </div>
      )}

      {sublabel && (
        <p className="text-xs text-[#5B4E75] font-medium text-center mt-2 max-w-[200px] leading-relaxed">
          {sublabel}
        </p>
      )}
    </div>
  );
}
