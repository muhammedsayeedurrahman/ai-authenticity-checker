import React from 'react';
import { AlertTriangle, CheckCircle, ShieldAlert } from 'lucide-react';

export default function VerdictCard({ verdict }) {
  if (!verdict) return null;

  const vUpper = verdict.toUpperCase();
  let color = '#22C55E';
  let bgColor = 'rgba(34,197,94,0.1)';
  let borderColor = 'rgba(34,197,94,0.3)';
  let Icon = CheckCircle;

  if (vUpper.includes('CRITICAL') || vUpper.includes('HIGH')) {
    color = '#EF4444';
    bgColor = 'rgba(239,68,68,0.1)';
    borderColor = 'rgba(239,68,68,0.3)';
    Icon = ShieldAlert;
  } else if (vUpper.includes('MEDIUM')) {
    color = '#EAB308';
    bgColor = 'rgba(234,179,8,0.1)';
    borderColor = 'rgba(234,179,8,0.3)';
    Icon = AlertTriangle;
  }

  return (
    <div
      className="p-4 rounded-xl border mt-4"
      style={{ backgroundColor: bgColor, borderColor }}
    >
      <div className="flex items-center gap-2 mb-2">
        <Icon size={20} style={{ color }} />
        <span className="font-bold text-sm tracking-wider" style={{ color }}>VERDICT</span>
      </div>
      <div className="text-text-primary text-sm font-medium leading-relaxed">
        {verdict}
      </div>
    </div>
  );
}
