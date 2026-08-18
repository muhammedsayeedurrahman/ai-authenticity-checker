import React from 'react';
import { AlertTriangle, CheckCircle, ShieldAlert } from 'lucide-react';
import { classifyVerdict } from '../utils/risk';

const TIERS = {
  critical: {
    Icon: ShieldAlert,
    bgClass: 'bg-risk-criticalDim border-[rgba(251,113,133,0.15)]',
    accentClass: 'text-risk-critical',
  },
  caution: {
    Icon: AlertTriangle,
    bgClass: 'bg-risk-cautionDim border-[rgba(250,204,21,0.15)]',
    accentClass: 'text-risk-caution',
  },
  clear: {
    Icon: CheckCircle,
    bgClass: 'bg-risk-clearDim border-[rgba(34,197,94,0.15)]',
    accentClass: 'text-risk-clear',
  },
};

export default function VerdictCard({ verdict, riskScore }) {
  if (!verdict) return null;

  const tier = classifyVerdict(verdict, riskScore);
  const { Icon, bgClass, accentClass } = TIERS[tier];

  return (
    <div className={`rounded-lg p-4 mt-3 border ${bgClass}`}>
      <div className={`flex items-center gap-2 mb-1.5 ${accentClass}`}>
        <Icon size={14} />
        <span className="label-tag">Verdict</span>
      </div>
      <p className="text-sm font-medium leading-snug text-text-1">
        {verdict}
      </p>
    </div>
  );
}
