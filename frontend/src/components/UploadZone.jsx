import React, { useCallback, useState, useRef, useEffect } from 'react';
import { UploadCloud, Check, RefreshCw, AlertCircle, FolderOpen, X } from 'lucide-react';
import { isFileAccepted } from '../utils/format';

const FORMAT_LABELS = {
  'image/*': ['JPG', 'PNG', 'WEBP'],
  'video/*': ['MP4', 'AVI', 'MOV'],
  'audio/*': ['WAV', 'MP3', 'FLAC'],
};

export default function UploadZone({
  onFileSelect,
  accept = 'image/*',
  label = 'Drag & drop or click to browse',
  initialFile = null,
  maxSizeMB = 500,
}) {
  const [isDragActive, setIsDragActive] = useState(false);
  const [preview, setPreview] = useState(null);
  const [rejected, setRejected] = useState(false);
  const [sizeError, setSizeError] = useState(false);
  const blobUrlRef = useRef(null);
  const inputRef = useRef(null);

  const badges = FORMAT_LABELS[accept] || ['JPG', 'PNG', 'MP4', 'WAV'];

  // Clean up blob URL on unmount or when preview changes
  useEffect(() => {
    return () => {
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, []);

  const handleFile = useCallback(
    (file) => {
      setRejected(false);
      setSizeError(false);

      // Client-side type validation
      if (!isFileAccepted(file, accept)) {
        setRejected(true);
        setTimeout(() => setRejected(false), 3000);
        return;
      }

      // Client-side size validation
      if (maxSizeMB > 0 && file.size > maxSizeMB * 1024 * 1024) {
        setSizeError(true);
        setTimeout(() => setSizeError(false), 4000);
        return;
      }

      // Revoke previous blob URL
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }

      onFileSelect(file);

      if (file.type.startsWith('image/') || file.type.startsWith('video/')) {
        const url = URL.createObjectURL(file);
        blobUrlRef.current = url;
        setPreview({ url, type: file.type, name: file.name });
      } else {
        setPreview({ name: file.name, type: file.type });
      }
    },
    [accept, onFileSelect, maxSizeMB],
  );

  // Accept an initial file from drag-drop on Dashboard
  useEffect(() => {
    if (initialFile) handleFile(initialFile);
  }, [initialFile, handleFile]);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragActive(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile],
  );

  const handleChange = useCallback(
    (e) => {
      const file = e.target.files[0];
      if (file) handleFile(file);
      // Reset so re-selecting the same file still fires onChange next time
      e.target.value = '';
    },
    [handleFile],
  );

  const handleBrowseClick = useCallback(() => {
    inputRef.current?.click();
  }, []);

  const handleClear = useCallback(
    (e) => {
      e.stopPropagation();
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
      setPreview(null);
      setRejected(false);
      setSizeError(false);
      if (inputRef.current) inputRef.current.value = '';
      onFileSelect(null);
    },
    [onFileSelect],
  );

  return (
    <div className="w-full space-y-2">
      <div
        onDragOver={(e) => { e.preventDefault(); setIsDragActive(true); }}
        onDragLeave={() => setIsDragActive(false)}
        onDrop={handleDrop}
        onClick={() => { if (!preview) handleBrowseClick(); }}
        className={`relative w-full rounded-xl flex flex-col items-center justify-center overflow-hidden min-h-[200px] border-2 border-dashed transition-colors duration-200 ${
          preview ? '' : 'cursor-pointer'
        } ${
          rejected || sizeError
            ? 'border-risk-critical bg-risk-criticalDim'
            : isDragActive
              ? 'border-accent bg-[rgba(59,130,246,0.05)]'
              : 'border-border-mid bg-bg-inset'
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={accept}
          onChange={handleChange}
          className="hidden"
          aria-label={label}
        />

        {/* HUD corner brackets — viewfinder accent, reinforces "scanning" affordance */}
        <span className={`hud-corner z-20 top-2.5 left-2.5 border-t-2 border-l-2 rounded-tl-md ${isDragActive ? 'border-accent' : ''}`} aria-hidden="true" />
        <span className={`hud-corner z-20 top-2.5 right-2.5 border-t-2 border-r-2 rounded-tr-md ${isDragActive ? 'border-accent' : ''}`} aria-hidden="true" />
        <span className={`hud-corner z-20 bottom-2.5 left-2.5 border-b-2 border-l-2 rounded-bl-md ${isDragActive ? 'border-accent' : ''}`} aria-hidden="true" />
        <span className={`hud-corner z-20 bottom-2.5 right-2.5 border-b-2 border-r-2 rounded-br-md ${isDragActive ? 'border-accent' : ''}`} aria-hidden="true" />

        {rejected || sizeError ? (
          <div className="flex flex-col items-center justify-center z-20 text-center px-6 py-6">
            <AlertCircle size={28} className="mb-2 text-risk-critical" />
            <p className="text-sm font-medium text-risk-critical">
              {sizeError ? 'File too large' : 'Unsupported file type'}
            </p>
            <p className="text-xs mt-1 text-text-2">
              {sizeError
                ? `Maximum file size is ${maxSizeMB} MB`
                : `Please select a ${accept.replace('/*', '')} file`}
            </p>
          </div>
        ) : preview && preview.url && preview.type?.startsWith('image/') ? (
          <>
            <img
              src={preview.url}
              alt={preview.name}
              className="absolute inset-0 w-full h-full object-cover"
            />
            <div className="absolute inset-0 bg-gradient-to-t from-black/75 via-black/5 to-black/10" />
            <div className="relative z-20 w-full flex flex-col items-center justify-end h-full text-center px-4 py-3">
              <div className="flex items-center gap-2 bg-black/50 backdrop-blur-sm px-3 py-1.5 rounded-lg max-w-full">
                <Check size={14} className="text-risk-clear flex-shrink-0" />
                <span className="text-sm font-medium truncate max-w-[200px] text-white">
                  {preview.name || 'File loaded'}
                </span>
              </div>
              <p className="text-xs mt-2 flex items-center gap-1 text-white/80">
                <RefreshCw size={10} /> Drop new file to replace
              </p>
            </div>
          </>
        ) : preview ? (
          <div className="flex flex-col items-center justify-center z-20 text-center px-6 py-6">
            <div className="flex items-center gap-2">
              <Check size={14} className="text-risk-clear" />
              <span className="text-sm font-medium truncate max-w-[220px] text-text-1">
                {preview.name || 'File loaded'}
              </span>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center z-20 pointer-events-none text-center px-6 py-6">
            <UploadCloud
              size={30}
              className={`mb-3 ${isDragActive ? 'text-accent' : 'text-text-3'}`}
            />
            <p className="text-sm font-medium text-text-2">
              {label}
            </p>
            <p className="text-xs mt-1 text-text-3">
              Supports high-res forensic analysis
            </p>
            <div className="flex gap-1.5 mt-3">
              {badges.map((ext) => (
                <span
                  key={ext}
                  className="text-[10px] px-2 py-0.5 rounded font-mono bg-bg-elevated text-text-3 border border-border-dim"
                >
                  {ext}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Explicit upload / clear controls */}
      <div className="flex gap-2">
        <button
          type="button"
          onClick={handleBrowseClick}
          className="btn-ghost flex-1 py-2 text-xs"
        >
          <FolderOpen size={13} />
          {preview ? 'Replace File' : 'Upload'}
        </button>
        <button
          type="button"
          onClick={handleClear}
          disabled={!preview}
          className="btn-danger py-2 px-3 text-xs disabled:opacity-30 disabled:cursor-not-allowed"
        >
          <X size={13} />
          Clear
        </button>
      </div>
    </div>
  );
}
