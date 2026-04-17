import React, { useCallback, useState } from 'react';
import { UploadCloud } from 'lucide-react';

export default function UploadZone({ onFileSelect, accept = "image/*", label = "Drag & drop file here or click to browse" }) {
  const [isDragActive, setIsDragActive] = useState(false);
  const [preview, setPreview] = useState(null);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragActive(false);
    const file = e.dataTransfer.files[0];
    if (file) {
      handleFile(file);
    }
  }, []);

  const handleChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      handleFile(file);
    }
  };

  const handleFile = (file) => {
    onFileSelect(file);
    // Create preview if it's an image or video
    if (file.type.startsWith('image/') || file.type.startsWith('video/')) {
      const objectUrl = URL.createObjectURL(file);
      setPreview({ url: objectUrl, type: file.type });
    } else {
      setPreview({ name: file.name, type: file.type });
    }
  };

  return (
    <div
      onDragOver={(e) => { e.preventDefault(); setIsDragActive(true); }}
      onDragLeave={() => setIsDragActive(false)}
      onDrop={handleDrop}
      className={`relative w-full h-64 border-2 border-dashed rounded-2xl flex flex-col items-center justify-center transition-all duration-300 overflow-hidden cursor-pointer
        ${isDragActive 
          ? 'border-accent-cyan bg-[rgba(0,240,255,0.05)] shadow-[inset_0_0_20px_rgba(0,240,255,0.1)]' 
          : 'border-border-subtle hover:border-border-glow bg-background-card hover:bg-background-card-hover'
        }`}
    >
      <input 
        type="file" 
        accept={accept}
        onChange={handleChange} 
        className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
      />
      
      {preview ? (
        <div className="absolute inset-0 w-full h-full bg-background z-0">
          {preview.type.startsWith('image/') ? (
            <img src={preview.url} alt="Preview" className="w-full h-full object-contain opacity-40 blur-sm" />
          ) : preview.type.startsWith('video/') ? (
            <video src={preview.url} className="w-full h-full object-contain opacity-40 blur-sm" />
          ) : null}
          <div className="absolute inset-0 flex flex-col items-center justify-center p-4 text-center z-20">
            <div className="bg-background-card backdrop-blur-md rounded-xl p-4 border border-border-subtle shadow-lg">
              <p className="text-accent-cyan font-bold mb-1">File Added</p>
              <p className="text-sm text-text-primary truncate max-w-xs">{preview.name || "Media File"}</p>
              <p className="text-xs text-text-muted mt-2">Click or drag a new file to replace</p>
            </div>
          </div>
        </div>
      ) : (
        <div className="flex flex-col items-center justify-center z-20 pointer-events-none text-center px-4">
          <div className="w-16 h-16 rounded-full bg-[rgba(255,255,255,0.02)] border border-[rgba(255,255,255,0.05)] flex items-center justify-center mb-4 transition-transform duration-300 group-hover:scale-110">
            <UploadCloud size={32} className={isDragActive ? "text-accent-cyan" : "text-text-secondary"} />
          </div>
          <p className="text-text-primary font-medium">{label}</p>
          <p className="text-xs text-text-muted mt-2">Supports high-res forensic analysis</p>
        </div>
      )}
    </div>
  );
}
