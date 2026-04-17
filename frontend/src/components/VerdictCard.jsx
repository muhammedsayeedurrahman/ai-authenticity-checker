import React from 'react';
import { AlertTriangle, CheckCircle, ShieldAlert } from 'lucide-react';

export default function VerdictCard({ verdict }) {
  if (!verdict) return null;

  const vUpper = verdict.toUpperCase();
  let color = '#10B981'; // Default Low green
  let bgColor = 'rgba(16,185,129,0.1)';
  let borderColor = 'rgba(16,185,129,0.3)';
  let Icon = CheckCircle;

  if (vUpper.includes('CRITICAL') || vUpper.includes('HIGH')) {
    color = '#EC4899';
    bgColor = 'rgba(236,72,153,0.1)';
    borderColor = 'rgba(236,72,153,0.3)';
    Icon = ShieldAlert;
  } else if (vUpper.includes('MEDIUM')) {
    color = '#F59E0B';
    bgColor = 'rgba(245,158,11,0.1)';
    borderColor = 'rgba(245,158,11,0.3)';
    Icon = AlertTriangle;
  }

  return (
    <div 
      className="p-4 rounded-xl border mt-4" 
      style={{ backgroundColor: bgColor, borderColor: borderColor }}
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
