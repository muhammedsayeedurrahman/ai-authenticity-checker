import React from 'react';
import { getRiskColorRaw, getRiskBg, normalizeScore } from '../utils/risk';

/**
 * Compact risk-percentage badge used in list/table rows.
 * Shared between History and Dashboard so "risk in a row" has one visual definition.
 */
export default function RiskBadge({ score }) {
  const pct = normalizeScore(score);
  const color = getRiskColorRaw(pct);
  const bg = getRiskBg(pct);
  return (
    <span
      className="label-tag font-mono"
      style={{ color, background: bg }}
    >
      {pct.toFixed(1)}%
    </span>
  );
}
