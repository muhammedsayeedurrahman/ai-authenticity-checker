import React from 'react';

export default function FrameTable({ framesRawStr }) {
  if (!framesRawStr) return null;

  // The backend returns a pre-formatted string with a header, line, and rows.
  // We'll parse it into a structured array to construct a native HTML table.
  
  const lines = framesRawStr.split('\n');
  if (lines.length < 3) return <pre className="text-xs text-text-muted">{framesRawStr}</pre>;

  const rows = [];
  // Skip the first 2 lines (header and dashes)
  for (let i = 2; i < lines.length; i++) {
    const line = lines[i].trim();
    if (!line) continue;
    
    // Split by multiple spaces
    const cols = line.split(/\s+/);
    if (cols.length >= 8) {
      rows.push(cols);
    }
  }

  return (
    <div className="w-full overflow-x-auto rounded-xl border border-border-subtle bg-background-card">
      <table className="w-full text-left text-xs">
        <thead className="bg-[#121828] text-text-secondary uppercase tracking-wider text-[10px]">
          <tr>
            <th className="px-4 py-3">Frame</th>
            <th className="px-4 py-3">Time</th>
            <th className="px-4 py-3">Risk</th>
            <th className="px-4 py-3">Pred</th>
            <th className="px-4 py-3 text-center">Face</th>
            <th className="px-4 py-3">ViT</th>
            <th className="px-4 py-3">Freq</th>
            <th className="px-4 py-3">Forns</th>
            <th className="px-4 py-3">FaceM</th>
            <th className="px-4 py-3">DINO</th>
            <th className="px-4 py-3">Eff</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[rgba(255,255,255,0.04)]">
          {rows.map((row, idx) => {
            const risk = parseFloat(row[2]) || 0;
            const riskColor = risk > 0.70 ? '#EC4899' : risk > 0.40 ? '#F59E0B' : '#10B981';
            
            return (
              <tr key={idx} className="hover:bg-[rgba(255,255,255,0.02)] transition-colors">
                <td className="px-4 py-2 font-mono text-text-primary">{row[0]}</td>
                <td className="px-4 py-2 font-mono text-text-muted">{row[1]}</td>
                <td className="px-4 py-2 font-mono font-bold" style={{ color: riskColor }}>{row[2]}</td>
                <td className="px-4 py-2 text-[10px]">{row[3]}</td>
                <td className="px-4 py-2 text-center text-text-muted">{row[4]}</td>
                <td className="px-4 py-2 font-mono text-text-muted">{row[5]}</td>
                <td className="px-4 py-2 font-mono text-text-muted">{row[6]}</td>
                <td className="px-4 py-2 font-mono text-text-muted">{row[7]}</td>
                <td className="px-4 py-2 font-mono text-text-muted">{row[8]}</td>
                <td className="px-4 py-2 font-mono text-text-muted">{row[9]}</td>
                <td className="px-4 py-2 font-mono text-text-muted">{row[10]}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
