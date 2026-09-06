import React from 'react';
import { getRiskColorRaw, normalizeScore } from '../utils/risk';
import { parseFrameDetails } from '../utils/frameParser';

const COLUMNS = [
  { key: 0, label: 'Frame', title: 'Frame number' },
  { key: 1, label: 'Time',  title: 'Timestamp in video' },
  { key: 2, label: 'Risk',  title: 'Aggregated risk score', isRisk: true },
  { key: 3, label: 'Pred',  title: 'Prediction label' },
  { key: 4, label: 'Face',  title: 'Face detected', center: true },
  { key: 5, label: 'ViT',   title: 'Vision Transformer score' },
  { key: 6, label: 'Freq',  title: 'Frequency analysis score' },
  { key: 7, label: 'Forns', title: 'Forensics model score' },
  { key: 8, label: 'FaceM', title: 'Face manipulation score' },
  { key: 9, label: 'DINO',  title: 'DINOv2 model score' },
  { key: 10, label: 'Eff',  title: 'EfficientNet score' },
];

export default function FrameTable({ framesRawStr }) {
  if (!framesRawStr) return null;

  const rows = parseFrameDetails(framesRawStr);
  if (rows.length === 0) return <pre className="text-xs text-[#8F81A8]">{framesRawStr}</pre>;

  return (
    <div className="w-full rounded-2xl overflow-hidden border border-purple-100 bg-white shadow-sm">
      {/* Mobile: card layout */}
      <div className="md:hidden space-y-2 p-2">
        {rows.map((row, idx) => (
          <div key={idx} className="p-3 rounded-xl bg-purple-50/60 border border-purple-100">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-mono font-bold text-[#1E1238]">
                Frame {row[0]}
              </span>
              <span
                className="text-xs font-mono font-bold px-2 py-0.5 rounded-full"
                style={{
                  color: getRiskColorRaw(normalizeScore(row[2])),
                  background: `${getRiskColorRaw(normalizeScore(row[2]))}18`,
                }}
              >
                Risk: {row[2]}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
              <span className="text-[#8F81A8]">Time</span>
              <span className="font-mono text-[#1E1238] font-semibold">{row[1]}</span>
              <span className="text-[#8F81A8]">Prediction</span>
              <span className="text-[#1E1238] font-semibold">{row[3]}</span>
              <span className="text-[#8F81A8]">Face</span>
              <span className="text-[#1E1238] font-semibold">{row[4]}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Desktop: scrollable table */}
      <div className="hidden md:block table-scroll">
        <table className="w-full text-left text-xs min-w-[700px]">
          <thead>
            <tr className="bg-purple-100/60 border-b border-purple-200/70">
              {COLUMNS.map((col) => (
                <th
                  key={col.key}
                  title={col.title}
                  className={`px-3.5 py-3 text-[11px] font-bold uppercase tracking-wider text-purple-900 ${col.center ? 'text-center' : ''}`}
                >
                  {col.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-purple-100/70">
            {rows.map((row, idx) => (
              <tr key={idx} className="hover:bg-purple-50/60 transition-colors">
                {COLUMNS.map((col) => {
                  const value = row[col.key];
                  if (col.isRisk) {
                    return (
                      <td
                        key={col.key}
                        className="px-3.5 py-2.5 font-mono font-black"
                        style={{ color: getRiskColorRaw(normalizeScore(value)) }}
                      >
                        {value}
                      </td>
                    );
                  }
                  return (
                    <td
                      key={col.key}
                      className={`px-3.5 py-2.5 font-mono ${col.center ? 'text-center' : ''} ${col.key === 0 ? 'text-[#1E1238] font-bold' : 'text-[#5B4E75]'}`}
                    >
                      {value}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
