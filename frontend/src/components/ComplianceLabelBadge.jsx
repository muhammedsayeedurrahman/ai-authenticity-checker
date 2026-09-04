import React from 'react';
import { FileCheck2 } from 'lucide-react';

const CODE_STYLES = {
  synthetically_generated: {
    wrap: 'bg-risk-criticalDim border-[rgba(251,113,133,0.15)]',
    text: 'text-risk-critical',
  },
  possibly_synthetic: {
    wrap: 'bg-risk-cautionDim border-[rgba(250,204,21,0.15)]',
    text: 'text-risk-caution',
  },
  indeterminate: {
    wrap: 'bg-white/[0.03] border-border-dim',
    text: 'text-text-3',
  },
};

/**
 * India IT Rules 2026 compliance-labeling badge.
 *
 * Renders nothing for "no_synthetic_indicators" or an absent label — this
 * surfaces the labeling/traceability determination from
 * core/compliance_label.py, not a replacement for the verdict/risk display.
 */
export default function ComplianceLabelBadge({ label }) {
  if (!label || !label.label_code || label.label_code === 'no_synthetic_indicators') return null;

  const style = CODE_STYLES[label.label_code] || CODE_STYLES.indeterminate;

  return (
    <div className={`rounded-lg p-4 mt-3 border ${style.wrap}`}>
      <div className={`flex items-center gap-2 mb-1.5 ${style.text}`}>
        <FileCheck2 size={14} />
        <span className="label-tag">Compliance Label — India IT Rules 2026</span>
      </div>
      <p className="text-sm font-semibold leading-snug text-text-1">
        {label.label_display}
      </p>
      {label.label_basis?.length > 0 && (
        <ul className="mt-2 space-y-0.5">
          {label.label_basis.map((basis, i) => (
            <li key={i} className="text-xs text-text-3 flex gap-1.5">
              <span className={style.text}>&bull;</span>
              {basis}
            </li>
          ))}
        </ul>
      )}
      <div className="flex flex-wrap gap-3 mt-2 text-xs text-text-2">
        {label.requires_visible_label && <span>Requires visible label</span>}
        {label.sla_applies && <span>3-hour takedown SLA applies</span>}
      </div>
      {label.disclaimer && (
        <p className="text-[10px] mt-2 italic text-text-3">
          {label.disclaimer}
        </p>
      )}
    </div>
  );
}
