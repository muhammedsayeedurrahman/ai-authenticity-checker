import React from 'react';
import { AlertTriangle, CheckCircle, ShieldAlert } from 'lucide-react';
import { classifyVerdict } from '../utils/risk';

const TIERS = {
  critical: { Icon: ShieldAlert,  className: 'bg-risk-criticalDim text-risk-critical border-[rgba(251,113,133,0.20)]' },
  caution:  { Icon: AlertTriangle, className: 'bg-risk-cautionDim text-risk-caution border-[rgba(250,204,21,0.20)]' },
  clear:    { Icon: CheckCircle,  className: 'bg-risk-clearDim text-risk-clear border-[rgba(34,197,94,0.20)]' },
};

/** Compact status pill for table rows — reuses VerdictCard's tier classification. */
export default function VerdictChip({ verdict, riskScore }) {
  if (!verdict) return null;

  const tier = classifyVerdict(verdict, riskScore);
  const { Icon, className } = TIERS[tier];

  return (
    <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full border text-[11px] font-medium truncate max-w-[140px] ${className}`}>
      <Icon size={10} className="flex-shrink-0" />
      <span className="truncate">{verdict}</span>
    </span>
  );
}
