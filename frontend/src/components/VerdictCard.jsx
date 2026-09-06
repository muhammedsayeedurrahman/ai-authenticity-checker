import React from 'react';
import { motion } from 'framer-motion';
import { AlertTriangle, CheckCircle, ShieldAlert } from 'lucide-react';
import { classifyVerdict } from '../utils/risk';

const TIERS = {
  critical: {
    Icon: ShieldAlert,
    bgClass: 'bg-risk-criticalDim border-[rgba(225,29,72,0.18)]',
    accentClass: 'text-risk-critical',
    barColor: 'var(--risk-critical)',
    pulse: true,
  },
  caution: {
    Icon: AlertTriangle,
    bgClass: 'bg-risk-cautionDim border-[rgba(245,158,11,0.18)]',
    accentClass: 'text-risk-caution',
    barColor: 'var(--risk-caution)',
    pulse: false,
  },
  clear: {
    Icon: CheckCircle,
    bgClass: 'bg-risk-clearDim border-[rgba(16,185,129,0.18)]',
    accentClass: 'text-risk-clear',
    barColor: 'var(--risk-clear)',
    pulse: false,
  },
};

export default function VerdictCard({ verdict, riskScore }) {
  if (!verdict) return null;

  const tier = classifyVerdict(verdict, riskScore);
  const { Icon, bgClass, accentClass, barColor, pulse } = TIERS[tier];

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96, y: 6 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      className={`relative overflow-hidden rounded-lg p-4 mt-3 border ${bgClass}`}
    >
      {/* Accent bar */}
      <span
        className="absolute left-0 top-0 bottom-0 w-1"
        style={{ background: barColor }}
      />

      <div className={`flex items-center gap-2 mb-1.5 pl-2 ${accentClass}`}>
        <motion.span
          animate={pulse ? { scale: [1, 1.15, 1] } : {}}
          transition={pulse ? { duration: 1.6, repeat: Infinity, ease: 'easeInOut' } : {}}
          className="flex items-center justify-center"
        >
          <Icon size={14} />
        </motion.span>
        <span className="label-tag">Verdict</span>
      </div>
      <p className="text-sm font-medium leading-snug text-text-1 pl-2">
        {verdict}
      </p>
    </motion.div>
  );
}
