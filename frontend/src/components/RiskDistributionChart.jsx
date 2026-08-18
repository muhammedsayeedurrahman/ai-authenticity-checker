import React from 'react';
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from 'recharts';
import { PieChart as PieChartIcon } from 'lucide-react';
import AnimatedNumber from './AnimatedNumber';

const TIER_COLORS = { clear: '#22C55E', caution: '#FACC15', critical: '#FB7185' };
const TIER_LABELS = { clear: 'cleared', caution: 'caution', critical: 'critical' };

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const { name, value } = payload[0];
  return (
    <div className="inset-panel px-3 py-2 text-xs">
      <p className="font-mono font-bold text-text-1">{value} {name}</p>
    </div>
  );
}

export default function RiskDistributionChart({ counts, total }) {
  const data = ['clear', 'caution', 'critical']
    .map((tier) => ({ tier, name: TIER_LABELS[tier], value: counts[tier] || 0 }))
    .filter((d) => d.value > 0);

  return (
    <div className="card h-full flex flex-col">
      <div className="flex items-center gap-2 mb-3">
        <PieChartIcon size={13} className="text-accent" />
        <span className="label-tag">Risk Distribution</span>
      </div>

      {total === 0 ? (
        <div className="flex-1 flex items-center justify-center text-center text-text-3 text-sm">
          No scans yet
        </div>
      ) : (
        <>
          <div className="relative h-[160px]">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={data}
                  dataKey="value"
                  nameKey="name"
                  innerRadius="65%"
                  outerRadius="88%"
                  paddingAngle={2}
                  animationDuration={800}
                  animationEasing="ease-out"
                >
                  {data.map((d) => (
                    <Cell key={d.tier} fill={TIER_COLORS[d.tier]} stroke="none" />
                  ))}
                </Pie>
                <Tooltip content={<ChartTooltip />} />
              </PieChart>
            </ResponsiveContainer>

            {/* Center total overlay */}
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
              <span className="font-display text-xl font-bold text-text-1">
                <AnimatedNumber value={total} />
              </span>
              <span className="text-[10px] uppercase tracking-wide text-text-3">Total</span>
            </div>
          </div>

          <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 justify-center">
            {['clear', 'caution', 'critical'].map((tier) => (
              <span key={tier} className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full" style={{ background: TIER_COLORS[tier] }} />
                <span className="text-xs font-mono text-text-3">{counts[tier] || 0} {TIER_LABELS[tier]}</span>
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
