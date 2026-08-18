import React from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { TrendingUp } from 'lucide-react';
import { normalizeScore, getRiskColorRaw } from '../utils/risk';
import { formatShortDateTime } from '../utils/format';

const AXIS_TICK = { fill: '#6B7585', fontSize: 11 };

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const { risk, time } = payload[0].payload;
  return (
    <div className="inset-panel px-3 py-2 text-xs">
      <p className="text-text-3 mb-0.5">{time}</p>
      <p className="font-mono font-bold" style={{ color: getRiskColorRaw(risk) }}>
        {risk.toFixed(1)}% risk
      </p>
    </div>
  );
}

export default function RiskTrendChart({ history = [] }) {
  // History arrives newest-first from the API — reverse for a
  // left-to-right chronological trend line.
  const data = [...history]
    .reverse()
    .map((item) => ({
      time: formatShortDateTime(item.timestamp),
      risk: normalizeScore(item.risk_score),
    }));

  return (
    <div className="card h-full">
      <div className="flex items-center gap-2 mb-3">
        <TrendingUp size={13} className="text-accent" />
        <span className="label-tag">Risk Trend</span>
      </div>

      {data.length < 2 ? (
        <div className="h-[200px] flex items-center justify-center text-center text-text-3 text-sm">
          Not enough data yet for a trend
        </div>
      ) : (
        <div className="h-[200px]">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
              <defs>
                <linearGradient id="riskTrendFill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3B82F6" stopOpacity={0.35} />
                  <stop offset="100%" stopColor="#3B82F6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
              <XAxis dataKey="time" tick={AXIS_TICK} axisLine={false} tickLine={false} minTickGap={24} />
              <YAxis domain={[0, 100]} tick={AXIS_TICK} axisLine={false} tickLine={false} width={32} />
              <Tooltip content={<ChartTooltip />} cursor={{ stroke: 'rgba(59,130,246,0.25)' }} />
              <Area
                type="monotone"
                dataKey="risk"
                stroke="#3B82F6"
                strokeWidth={2}
                fill="url(#riskTrendFill)"
                animationDuration={800}
                animationEasing="ease-out"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
