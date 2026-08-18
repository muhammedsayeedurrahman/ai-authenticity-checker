import React from 'react';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { Activity } from 'lucide-react';
import { normalizeScore, getRiskColorRaw } from '../utils/risk';
import { parseFrameDetails } from '../utils/frameParser';

const AXIS_TICK = { fill: '#6B7585', fontSize: 11 };

function ChartTooltip({ active, payload }) {
  if (!active || !payload?.length) return null;
  const { risk, time, frame } = payload[0].payload;
  return (
    <div className="inset-panel px-3 py-2 text-xs">
      <p className="text-text-3 mb-0.5">Frame {frame} &middot; {time}</p>
      <p className="font-mono font-bold" style={{ color: getRiskColorRaw(risk) }}>
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
    <div className="card">
      <div className="flex items-center gap-2 mb-3">
        <Activity size={13} className="text-accent" />
        <span className="label-tag">Risk Timeline</span>
      </div>

      <div className="h-[180px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 8, right: 8, left: -16, bottom: 0 }}>
            <defs>
              <linearGradient id="videoTimelineFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#3B82F6" stopOpacity={0.35} />
                <stop offset="100%" stopColor="#3B82F6" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid stroke="rgba(255,255,255,0.06)" vertical={false} />
            <XAxis
              dataKey="frame"
              tick={AXIS_TICK}
              axisLine={false}
              tickLine={false}
              minTickGap={24}
              label={{ value: 'Frame', position: 'insideBottom', offset: -2, fill: '#6B7585', fontSize: 10 }}
            />
            <YAxis domain={[0, 100]} tick={AXIS_TICK} axisLine={false} tickLine={false} width={32} />
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: 'rgba(59,130,246,0.25)' }} />
            <Area
              type="monotone"
              dataKey="risk"
              stroke="#3B82F6"
              strokeWidth={2}
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
