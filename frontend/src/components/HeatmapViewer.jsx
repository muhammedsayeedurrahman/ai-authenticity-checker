import React, { useState } from 'react';
import { Layers } from 'lucide-react';

export default function HeatmapViewer({ originalFile, gradcamBase64 }) {
  const [showHeatmap, setShowHeatmap] = useState(true);

  if (!originalFile && !gradcamBase64) {
    return (
      <div className="w-full h-full min-h-[300px] flex items-center justify-center bg-[rgba(255,255,255,0.02)] rounded-xl border border-border-subtle">
        <p className="text-text-muted">Awaiting visual results</p>
      </div>
    );
  }

  const originalUrl = originalFile ? URL.createObjectURL(originalFile) : null;
  const heatmapUrl = gradcamBase64 ? `data:image/png;base64,${gradcamBase64}` : null;

  return (
    <div className="relative w-full h-full rounded-xl overflow-hidden glass-card">
      {/* Tool bar */}
      <div className="absolute top-4 right-4 z-10 flex gap-2">
        {heatmapUrl && (
          <button 
            onClick={() => setShowHeatmap(!showHeatmap)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold tracking-wide backdrop-blur-md transition-all border
              ${showHeatmap 
                ? 'bg-[rgba(236,72,153,0.2)] text-accent-pink border-[rgba(236,72,153,0.4)] shadow-[0_0_10px_rgba(236,72,153,0.2)]' 
                : 'bg-background-card text-text-primary border-border-subtle'
              }`}
          >
            <Layers size={14} />
            {showHeatmap ? 'HEATMAP ON' : 'HEATMAP OFF'}
          </button>
        )}
      </div>

      <div className="w-full h-full min-h-[400px] max-h-[600px] flex items-center justify-center p-2">
        <img 
          src={(showHeatmap && heatmapUrl) ? heatmapUrl : originalUrl} 
          alt="Analysis Output" 
          className="max-w-full max-h-full object-contain rounded-lg"
        />
      </div>
    </div>
  );
}
