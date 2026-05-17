import React, { useState } from 'react';
import { Eye, EyeOff } from 'lucide-react';

export default function HeatmapViewer({ originalFile, gradcamBase64 }) {
  const [showHeatmap, setShowHeatmap] = useState(true);

  const originalUrl = originalFile ? URL.createObjectURL(originalFile) : null;
  const heatmapUrl  = gradcamBase64 ? `data:image/png;base64,${gradcamBase64}` : null;

  if (!originalFile && !gradcamBase64) {
    return (
      <div
        className="w-full flex flex-col items-center justify-center rounded-[10px]"
        style={{
          minHeight: '380px',
          background: 'var(--bg-inset, #0A0C12)',
          border: '1px solid var(--border-dim, rgba(255,255,255,0.07))',
        }}
      >
        <Eye size={20} style={{ color: 'var(--text-3, #4A5264)', marginBottom: '8px' }} />
        <p className="text-[12px]" style={{ color: 'var(--text-3, #4A5264)' }}>
          Heatmap will appear here
        </p>
      </div>
    );
  }

  return (
    <div
      className="relative w-full rounded-[10px] overflow-hidden"
      style={{
        background: 'var(--bg-inset, #0A0C12)',
        border: '1px solid var(--border-dim, rgba(255,255,255,0.07))',
      }}
    >
      {/* Toggle */}
      {heatmapUrl && (
        <div className="absolute top-3 right-3 z-10">
          <button
            onClick={() => setShowHeatmap(!showHeatmap)}
            aria-label={showHeatmap ? 'Show original image' : 'Show GradCAM heatmap'}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-medium tracking-wide transition-all"
            style={{
              background: showHeatmap
                ? 'rgba(6,182,212,0.10)'
                : 'var(--bg-elevated, #1C2130)',
              border: showHeatmap
                ? '1px solid rgba(6,182,212,0.25)'
                : '1px solid var(--border-dim, rgba(255,255,255,0.07))',
              color: showHeatmap
                ? 'var(--accent-2, #06B6D4)'
                : 'var(--text-2, #8B95A5)',
              boxShadow: showHeatmap
                ? '0 0 12px rgba(6,182,212,0.10)'
                : 'none',
            }}
          >
            {showHeatmap ? <EyeOff size={11} /> : <Eye size={11} />}
            {showHeatmap ? 'GradCAM' : 'Original'}
          </button>
        </div>
      )}

      {/* Image */}
      <div className="w-full flex items-center justify-center p-3" style={{ minHeight: '380px', maxHeight: '580px' }}>
        <img
          src={(showHeatmap && heatmapUrl) ? heatmapUrl : originalUrl}
          alt="Analysis Output"
          className="max-w-full max-h-[560px] object-contain rounded-lg"
          style={{ transition: 'opacity 0.3s ease' }}
        />
      </div>
    </div>
  );
}
