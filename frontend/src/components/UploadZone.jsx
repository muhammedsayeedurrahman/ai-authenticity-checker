import React, { useCallback, useState } from 'react';
import { UploadCloud, Check, RefreshCw } from 'lucide-react';

const FORMAT_BADGES = ['JPG', 'PNG', 'MP4', 'WAV', 'MP3'];

export default function UploadZone({
  onFileSelect,
  accept = 'image/*',
  label = 'Drag & drop or click to browse',
}) {
  const [isDragActive, setIsDragActive] = useState(false);
  const [preview, setPreview] = useState(null);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragActive(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    []
  );

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
  };

  const handleFile = (file) => {
    onFileSelect(file);
    if (file.type.startsWith('image/') || file.type.startsWith('video/')) {
      setPreview({ url: URL.createObjectURL(file), type: file.type, name: file.name });
    } else {
      setPreview({ name: file.name, type: file.type });
    }
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragActive(true); }}
      onDragLeave={() => setIsDragActive(false)}
      onDrop={handleDrop}
      className="relative w-full h-56 rounded-lg flex flex-col items-center justify-center overflow-hidden cursor-pointer"
      style={{
        border: `2px dashed ${isDragActive ? 'var(--accent, #3B82F6)' : 'var(--border-mid, rgba(255,255,255,0.12))'}`,
        background: isDragActive
          ? 'rgba(59,130,246,0.05)'
          : 'linear-gradient(180deg, var(--bg-inset, #0A0C12) 0%, var(--bg-card, #141820) 100%)',
        boxShadow: isDragActive
          ? 'inset 0 0 30px rgba(59,130,246,0.12)'
          : '0 0 8px rgba(59,130,246,0.10)',
        animation: isDragActive ? 'none' : 'borderGlow 3s ease-in-out infinite',
        transition: 'border-color 0.2s ease, background 0.2s ease',
      }}
    >
      <input
        type="file"
        accept={accept}
        onChange={handleChange}
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
      />

      {preview ? (
        <div className="flex flex-col items-center justify-center z-20 text-center px-6">
          <div className="flex items-center gap-2">
            <Check size={14} style={{ color: 'var(--risk-clear, #34D399)' }} />
            <span
              className="text-[12px] font-medium truncate max-w-[200px]"
              style={{ color: 'var(--text-1, #EDF0F7)' }}
            >
              {preview.name || 'File loaded'}
            </span>
          </div>
          <p
            className="text-[10px] mt-1.5 flex items-center gap-1"
            style={{ color: 'var(--text-3, #4A5264)' }}
          >
            <RefreshCw size={9} /> Drop new file to replace
          </p>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center z-20 pointer-events-none text-center px-6">
          <UploadCloud
            size={32}
            style={{
              color: isDragActive
                ? 'var(--accent, #3B82F6)'
                : 'var(--text-3, #4A5264)',
              marginBottom: '12px',
              animation: 'breathe 3s ease-in-out infinite',
            }}
          />
          <p className="text-[12px] font-medium" style={{ color: 'var(--text-2, #8B95A5)' }}>
            {label}
          </p>
          <p className="text-[10px] mt-1" style={{ color: 'var(--text-3, #4A5264)' }}>
            Supports high-res forensic analysis
          </p>
          <div className="flex gap-1.5 mt-3">
            {FORMAT_BADGES.map((ext) => (
              <span
                key={ext}
                className="text-[9px] px-1.5 py-0.5 rounded font-mono"
                style={{
                  background: 'var(--bg-elevated, #1C2130)',
                  color: 'var(--text-3)',
                  border: '1px solid var(--border-dim)',
                }}
              >
                {ext}
              </span>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
