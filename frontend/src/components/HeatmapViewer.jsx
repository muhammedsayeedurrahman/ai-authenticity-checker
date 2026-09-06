import React, { useState, useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { Eye, EyeOff } from 'lucide-react';

export default function HeatmapViewer({ originalFile, gradcamBase64 }) {
  const [showHeatmap, setShowHeatmap] = useState(true);
  const blobUrlRef = useRef(null);

  // Create and clean up blob URL for the original file
  useEffect(() => {
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }
    if (originalFile) {
      blobUrlRef.current = URL.createObjectURL(originalFile);
    }
    return () => {
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, [originalFile]);

  const originalUrl = blobUrlRef.current;
  const heatmapUrl = gradcamBase64
    ? (gradcamBase64.startsWith('data:') ? gradcamBase64 : `data:image/png;base64,${gradcamBase64}`)
    : null;

  // Only reveal once there's an actual analysis result to show — the raw
  // uploaded file is already previewed inline in the UploadZone dropzone,
  // so showing it again here (pre-analysis) was a duplicate.
  if (!gradcamBase64) {
    return (
      <div className="w-full flex flex-col items-center justify-center rounded-xl min-h-[360px] bg-bg-inset border border-border-dim">
        <Eye size={22} className="mb-2 text-text-3" />
        <p className="text-sm text-text-3">
          Heatmap will appear here after analysis
        </p>
      </div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.98 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
      className="relative w-full rounded-xl overflow-hidden bg-bg-inset border border-border-dim"
    >
      {/* Toggle button */}
      {heatmapUrl && (
        <div className="absolute top-3 right-3 z-10">
          <button
            onClick={() => setShowHeatmap((prev) => !prev)}
            aria-label={showHeatmap ? 'Show original image' : 'Show GradCAM heatmap'}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium tracking-wide transition-all border ${
              showHeatmap
                ? 'bg-[rgba(56,189,248,0.10)] border-[rgba(56,189,248,0.25)] text-accent-2'
                : 'bg-bg-elevated border-border-dim text-text-2'
            }`}
          >
            {showHeatmap ? <EyeOff size={12} /> : <Eye size={12} />}
            {showHeatmap ? 'GradCAM' : 'Original'}
          </button>
        </div>
      )}

      {/* Image */}
      <div className="w-full flex items-center justify-center p-4 min-h-[360px] max-h-[560px]">
        <img
          src={showHeatmap && heatmapUrl ? heatmapUrl : originalUrl}
          alt={showHeatmap && heatmapUrl ? 'GradCAM heatmap overlay' : 'Uploaded image'}
          className="max-w-full max-h-[540px] object-contain rounded-lg transition-opacity duration-300"
        />
      </div>
    </motion.div>
  );
}
