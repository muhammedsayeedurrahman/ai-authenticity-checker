import React from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { Activity } from 'lucide-react';
import { normalizeScore, getRiskColorRaw } from '../utils/risk';
import { parseFrameDetails } from '../utils/frameParser';

const AXIS_TICK = { fill: '#8F81A8', fontSize: 11, fontWeight: 600 };

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const { risk, time, frame } = payload[0].payload;
  return (
    <div className="p-2.5 rounded-xl bg-white border border-purple-200 shadow-md text-xs">
      <p className="text-[#8F81A8] font-semibold mb-0.5">Frame {frame} &middot; {time}</p>
      <p className="font-mono font-black" style={{ color: getRiskColorRaw(risk) }}>
        {risk.toFixed(1)}% risk
      </p>
    </div>
  );
}

export default function VideoRiskTimeline({ framesRawStr }) {
  const rows = parseFrameDetails(framesRawStr);
  if (rows.length === 0) return null;

  const data = rows.map((row) => ({
    frame: row[0],
    time: row[1],
    risk: normalizeScore(row[2]),
  }));

  return (
    <div className="card space-y-3">
      <div className="flex items-center gap-2">
        <Activity size={14} className="text-purple-700" />
        <span className="label-tag">Temporal Risk Curve</span>
      </div>

      <div className="h-[180px] bg-white p-2 rounded-2xl border border-purple-100 shadow-sm">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
            <defs>
              <linearGradient id="videoTimelineFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#7C3AED" stopOpacity={0.4} />
                <stop offset="100%" stopColor="#7C3AED" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(139, 92, 246, 0.12)" vertical={false} />
            <XAxis
              dataKey="frame"
              tick={AXIS_TICK}
              axisLine={false}
              tickLine={false}
              minTickGap={24}
              label={{ value: 'Frame', position: 'insideBottom', offset: -2, fill: '#8F81A8', fontSize: 10, fontWeight: 700 }}
            />
            <YAxis domain={[0, 100]} tick={AXIS_TICK} axisLine={false} tickLine={false} width={32} />
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: '#7C3AED' }} />
            <Area
              type="monotone"
              dataKey="risk"
              stroke="#7C3AED"
              strokeWidth={2.5}
              fill="url(#videoTimelineFill)"
              animationDuration={800}
              animationEasing="ease-out"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
