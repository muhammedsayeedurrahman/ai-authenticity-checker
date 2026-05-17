import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Mic, Volume2, ShieldCheck } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import RiskGauge from '../components/RiskGauge';
import VerdictCard from '../components/VerdictCard';
import useForensicStore from '../store/useForensicStore';

const fadeUp = {
  hidden:  { opacity: 0, y: 14 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.22, 1, 0.36, 1] } },
};

export default function AudioAnalysis() {
  const [file, setFile] = useState(null);

  const { audioAnalysis, runAudioAnalysis } = useForensicStore();
  const { isAnalyzing, results, error } = audioAnalysis;

  const handleAnalyze = () => {
    if (!file) return;
    runAudioAnalysis(file);
  };

  return (
    <motion.div initial="hidden" animate="visible" variants={fadeUp} className="space-y-6">
      {/* Header */}
      <header>
        <div className="flex items-center gap-2.5 mb-1.5">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center"
            style={{ background: 'var(--accent-dim)', border: '1px solid rgba(59,130,246,0.18)', boxShadow: '0 0 12px rgba(59,130,246,0.15)' }}
          >
            <Mic size={14} style={{ color: 'var(--accent)' }} />
          </div>
          <h1 className="font-display text-xl font-bold" style={{ color: 'var(--text-1)' }}>Audio Analysis</h1>
        </div>
        <p className="text-[12px]" style={{ color: 'var(--text-2)' }}>
          Voice cloning and synthetic speech detection using frequency analysis.
        </p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        {/* Left: Upload + button */}
        <div className="space-y-4">
          <UploadZone onFileSelect={setFile} accept="audio/*" label="Upload Audio (WAV/MP3)" />

          <button
            onClick={handleAnalyze}
            disabled={!file || isAnalyzing}
            className="btn-primary w-full py-3 text-[12px]"
          >
            {isAnalyzing ? (
              <>
                <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Processing Audio...
              </>
            ) : (
              <><Volume2 size={15} /> EXTRACT SPECTROGRAMS</>
            )}
          </button>
        </div>

        {/* Right: Results */}
        <div className="card min-h-[400px]">
          <div className="flex items-center gap-2 mb-4">
            <Mic size={13} style={{ color: 'var(--text-3)' }} />
            <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: 'var(--text-3)' }}>Spectral Intelligence</span>
          </div>

          {error ? (
            <div
              role="alert"
              className="p-3 rounded-lg text-[11px]"
              style={{ background: 'rgba(251,113,133,0.08)', border: '1px solid rgba(251,113,133,0.20)', color: 'var(--risk-critical)' }}
            >
              {error}
            </div>
          ) : results ? (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center space-y-4">
              <div className="flex gap-10">
                <RiskGauge percentage={results.risk_percentage || results.data?.risk_percent || 0} label="AI Risk" size={160} />
                <RiskGauge percentage={results.authenticity_percentage || results.data?.authenticity_score || 0} label="Authenticity" size={160} />
              </div>

              {results.risk_label && (
                <div
                  className="w-full text-center py-2 rounded-lg text-[11px] font-semibold tracking-wide"
                  style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-dim)', color: 'var(--text-2)' }}
                >
                  {results.risk_label}
                </div>
              )}

              <div className="w-full">
                <VerdictCard verdict={results.verdict || results.data?.verdict} />
              </div>

              {(results.details || results.data?.explanation) && (
                <div
                  className="w-full p-3 rounded-lg"
                  style={{ background: 'var(--bg-inset)', border: '1px solid var(--border-dim)' }}
                >
                  <pre className="text-[10px] font-mono whitespace-pre-wrap leading-relaxed" style={{ color: 'var(--text-2)' }}>
                    {results.details || results.data.explanation}
                  </pre>
                </div>
              )}
            </motion.div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center mt-20 space-y-2" style={{ color: 'var(--text-3)' }}>
              <ShieldCheck size={20} className="opacity-30" />
              <p className="text-[11px]">Waiting for audio input</p>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
