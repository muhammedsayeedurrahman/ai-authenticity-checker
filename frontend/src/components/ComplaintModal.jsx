import React, { useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { ShieldAlert, X } from 'lucide-react';
import ComplaintForm from './ComplaintForm';

/**
 * Modal wrapper around ComplaintForm, opened from an analysis results
 * panel (already has the analysis + file name in hand). See
 * pages/CyberComplaint.jsx for the standalone "pick a past analysis
 * first" version of the same form.
 */
export default function ComplaintModal({ open, onClose, analysis, fileName }) {
  useEffect(() => {
    if (!open) return;
    const handleKeyDown = (e) => { if (e.key === 'Escape') onClose(); };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open && (
        <div
          className="fixed inset-0 z-[100] flex items-center justify-center p-4"
          role="dialog"
          aria-modal="true"
          aria-labelledby="complaint-title"
        >
          <motion.div
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
            onClick={onClose}
            aria-hidden="true"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
          />

          <motion.div
            className="relative z-10 w-full max-w-md rounded-xl p-6 bg-bg-card border border-border-mid shadow-modal max-h-[90vh] overflow-y-auto"
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ duration: 0.2 }}
          >
            <div className="flex items-start justify-between gap-3 mb-4">
              <div className="flex items-start gap-3">
                <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5 border bg-risk-criticalDim border-[rgba(251,113,133,0.20)]">
                  <ShieldAlert size={16} className="text-risk-critical" />
                </div>
                <div>
                  <h2 id="complaint-title" className="text-sm font-semibold text-text-1">
                    Raise Cyber Crime Complaint
                  </h2>
                  <p className="text-xs mt-1 text-text-2 leading-relaxed">
                    Generates a complaint document for you to review and file yourself —
                    ProofyX does not submit anything on your behalf.
                  </p>
                </div>
              </div>
              <button
                type="button"
                onClick={onClose}
                className="p-1 rounded text-text-3 hover:text-text-1 hover:bg-white/5 flex-shrink-0"
                aria-label="Close"
              >
                <X size={15} />
              </button>
            </div>

            <ComplaintForm
              analysis={analysis}
              fileName={fileName}
              onDone={onClose}
              onCancel={onClose}
            />
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
}
