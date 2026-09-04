import React from 'react';
import { ShieldAlert } from 'lucide-react';

/**
 * Plain-language fraud/cybercrime advisory banner.
 *
 * Renders nothing when there's no flagged pattern (category is missing,
 * "none", or absent) — this is an additional signal layered on top of the
 * verdict, not a replacement for it. See core/cybercrime_risk.py for how
 * `risk` is derived.
 */
export default function CybercrimeRiskAdvisory({ risk }) {
  if (!risk || !risk.category || risk.category === 'none') return null;

  return (
    <div className="rounded-lg p-4 mt-3 border bg-risk-criticalDim border-[rgba(251,113,133,0.15)]">
      <div className="flex items-center gap-2 mb-1.5 text-risk-critical">
        <ShieldAlert size={14} />
        <span className="label-tag">Cybercrime Risk Advisory</span>
      </div>
      <p className="text-sm font-semibold leading-snug text-text-1">
        {risk.label}
      </p>
      {risk.description && (
        <p className="text-sm mt-1.5 leading-relaxed text-text-2">
          {risk.description}
        </p>
      )}
      {risk.signals?.length > 0 && (
        <ul className="mt-2 space-y-0.5">
          {risk.signals.map((signal, i) => (
            <li key={i} className="text-xs text-text-3 flex gap-1.5">
              <span className="text-risk-critical">&bull;</span>
              {signal}
            </li>
          ))}
        </ul>
      )}
      {risk.advisory && (
        <p className="text-xs mt-2 font-medium text-text-1">
          {risk.advisory}
        </p>
      )}
      {risk.disclaimer && (
        <p className="text-[10px] mt-2 italic text-text-3">
          {risk.disclaimer}
        </p>
      )}
    </div>
  );
}
