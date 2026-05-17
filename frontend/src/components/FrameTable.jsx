import React from 'react';

export default function FrameTable({ framesRawStr }) {
  if (!framesRawStr) return null;

  const lines = framesRawStr.split('\n');
  if (lines.length < 3) return <pre className="text-xs" style={{ color: 'var(--text-3, #4A5264)' }}>{framesRawStr}</pre>;

  const rows = [];
  for (let i = 2; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;

    const cols = line.split(/\s+/);
    if (cols.length >= 8) {
      rows.push(cols);
    }
  }

  const getRiskColor = (value) => {
    const risk = parseFloat(value) || 0;
    if (risk > 0.70) return '#FB7185';
    if (risk > 0.40) return '#FBBF24';
    return '#34D399';
  };

  const getRiskGlow = (value) => {
    const risk = parseFloat(value) || 0;
    if (risk > 0.70) return '0 0 6px rgba(251,113,133,0.3)';
    if (risk > 0.40) return '0 0 6px rgba(251,191,36,0.3)';
    return 'none';
  };

  return (
    <div
      className="card w-full overflow-x-auto rounded-[10px]"
      style={{
        background: 'var(--bg-card, #141820)',
        border: '1px solid var(--border-dim, rgba(255,255,255,0.07))',
      }}
    >
      <table className="w-full text-left text-xs">
        <thead>
          <tr style={{
            background: 'linear-gradient(135deg, rgba(59,130,246,0.08), rgba(6,182,212,0.04))',
            borderBottom: '1px solid rgba(59,130,246,0.10)',
          }}>
            <th className="px-3 py-2.5 text-[9px] uppercase tracking-wide" style={{ color: 'var(--text-3, #4A5264)' }}>Frame</th>
            <th className="px-3 py-2.5 text-[9px] uppercase tracking-wide" style={{ color: 'var(--text-3, #4A5264)' }}>Time</th>
            <th className="px-3 py-2.5 text-[9px] uppercase tracking-wide" style={{ color: 'var(--text-3, #4A5264)' }}>Risk</th>
            <th className="px-3 py-2.5 text-[9px] uppercase tracking-wide" style={{ color: 'var(--text-3, #4A5264)' }}>Pred</th>
            <th className="px-3 py-2.5 text-[9px] uppercase tracking-wide text-center" style={{ color: 'var(--text-3, #4A5264)' }}>Face</th>
            <th className="px-3 py-2.5 text-[9px] uppercase tracking-wide" style={{ color: 'var(--text-3, #4A5264)' }}>ViT</th>
            <th className="px-3 py-2.5 text-[9px] uppercase tracking-wide" style={{ color: 'var(--text-3, #4A5264)' }}>Freq</th>
            <th className="px-3 py-2.5 text-[9px] uppercase tracking-wide" style={{ color: 'var(--text-3, #4A5264)' }}>Forns</th>
            <th className="px-3 py-2.5 text-[9px] uppercase tracking-wide" style={{ color: 'var(--text-3, #4A5264)' }}>FaceM</th>
            <th className="px-3 py-2.5 text-[9px] uppercase tracking-wide" style={{ color: 'var(--text-3, #4A5264)' }}>DINO</th>
            <th className="px-3 py-2.5 text-[9px] uppercase tracking-wide" style={{ color: 'var(--text-3, #4A5264)' }}>Eff</th>
          </tr>
        </thead>
        <tbody className="divide-y" style={{ borderColor: 'rgba(255,255,255,0.04)' }}>
          {rows.map((row, idx) => (
            <tr
              key={idx}
              className="transition-colors"
              style={{ '--tw-divide-opacity': 1 }}
              onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(59,130,246,0.04)'}
              onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
            >
              <td className="px-3 py-2 font-mono" style={{ color: 'var(--text-1, #EDF0F7)' }}>{row[0]}</td>
              <td className="px-3 py-2 font-mono" style={{ color: 'var(--text-2, #8B95A5)' }}>{row[1]}</td>
              <td className="px-3 py-2 font-mono font-bold" style={{ color: getRiskColor(row[2]), textShadow: getRiskGlow(row[2]) }}>{row[2]}</td>
              <td className="px-3 py-2 text-[10px]" style={{ color: 'var(--text-2, #8B95A5)' }}>{row[3]}</td>
              <td className="px-3 py-2 text-center" style={{ color: 'var(--text-2, #8B95A5)' }}>{row[4]}</td>
              <td className="px-3 py-2 font-mono" style={{ color: 'var(--text-2, #8B95A5)' }}>{row[5]}</td>
              <td className="px-3 py-2 font-mono" style={{ color: 'var(--text-2, #8B95A5)' }}>{row[6]}</td>
              <td className="px-3 py-2 font-mono" style={{ color: 'var(--text-2, #8B95A5)' }}>{row[7]}</td>
              <td className="px-3 py-2 font-mono" style={{ color: 'var(--text-2, #8B95A5)' }}>{row[8]}</td>
              <td className="px-3 py-2 font-mono" style={{ color: 'var(--text-2, #8B95A5)' }}>{row[9]}</td>
              <td className="px-3 py-2 font-mono" style={{ color: 'var(--text-2, #8B95A5)' }}>{row[10]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
