import React from 'react';
import { AlertTriangle, CheckCircle, ShieldAlert } from 'lucide-react';

export default function VerdictCard({ verdict }) {
  if (!verdict) return null;

  const vUpper = verdict.toUpperCase();

  let color, Icon, glowColor;

  if (vUpper.includes('CRITICAL') || vUpper.includes('HIGH')) {
    color = '#FB7185';
    Icon = ShieldAlert;
    glowColor = 'rgba(251,113,133,0.12)';
  } else if (vUpper.includes('MEDIUM')) {
    color = '#FBBF24';
    Icon = AlertTriangle;
    glowColor = 'rgba(251,191,36,0.12)';
  } else {
    color = '#34D399';
    Icon = CheckCircle;
    glowColor = 'rgba(52,211,153,0.12)';
  }

  return (
    <div
      className="rounded-lg p-3 mt-3"
      style={{
        background: `${color}0F`,
        border: `1px solid ${color}26`,
        boxShadow: `0 0 16px ${glowColor}`,
      }}
    >
      <div className="flex items-center gap-2 mb-1">
        <Icon size={12} style={{ color }} />
        <span
          className="text-[9px] uppercase tracking-[0.12em] font-medium"
          style={{ color }}
        >
          Verdict
        </span>
      </div>
      <p
        className="text-[12px] font-medium leading-snug"
        style={{ color: 'var(--text-1, #EDF0F7)' }}
      >
        {verdict}
      </p>
    </div>
  );
}
