import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Layers, Zap, ShieldCheck } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import RiskGauge from '../components/RiskGauge';
import ScoreBar from '../components/ScoreBar';
import VerdictCard from '../components/VerdictCard';
import useForensicStore from '../store/useForensicStore';

const fadeUp = {
  hidden:  { opacity: 0, y: 14 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.22, 1, 0.36, 1] } },
};

export default function Multimodal() {
  const [image, setImage] = useState(null);
  const [video, setVideo] = useState(null);
  const [audio, setAudio] = useState(null);

  const { multimodalAnalysis, runMultimodalAnalysis } = useForensicStore();
  const { isAnalyzing, results, error } = multimodalAnalysis;

  const handleAnalyze = () => {
    if (!image && !video && !audio) return;
    runMultimodalAnalysis(image, video, audio);
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
            <Layers size={14} style={{ color: 'var(--accent)' }} />
          </div>
          <h1 className="font-display text-xl font-bold" style={{ color: 'var(--text-1)' }}>Multimodal Fusion</h1>
        </div>
        <p className="text-[12px]" style={{ color: 'var(--text-2)' }}>
          Upload multiple facets of media for a comprehensively compiled risk score.
        </p>
      </header>

      {/* 3-column upload grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <UploadZone onFileSelect={setImage} accept="image/*" label="Image (Optional)" />
        <UploadZone onFileSelect={setVideo} accept="video/*" label="Video (Optional)" />
        <UploadZone onFileSelect={setAudio} accept="audio/*" label="Audio (Optional)" />
      </div>

      {/* Analyze button */}
      <div className="flex justify-center">
        <button
          onClick={handleAnalyze}
          disabled={(!image && !video && !audio) || isAnalyzing}
          className="btn-primary max-w-sm w-full py-3 text-[12px]"
        >
          {isAnalyzing ? (
            <>
              <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Fusing Modalities...
            </>
          ) : (
            <><Zap size={15} /> INITIATE MULTIMODAL FUSION</>
          )}
        </button>
      </div>

      {/* Error */}
      {error && (
        <div
          role="alert"
          className="p-3 rounded-lg text-[12px] text-center"
          style={{ background: 'rgba(251,113,133,0.08)', border: '1px solid rgba(251,113,133,0.20)', color: 'var(--risk-critical)' }}
        >
          {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="card">
          <p className="text-[10px] font-bold uppercase tracking-widest mb-5 text-center" style={{ color: 'var(--text-3)' }}>
            Fusion Intelligence Report
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-center">
            {/* Left: Gauge + Verdict */}
            <div className="flex flex-col items-center space-y-4">
              <RiskGauge
                percentage={results.data?.risk_score || results.data?.risk_percent || results.risk_percentage || 0}
                label="Aggregated Risk"
                size={200}
              />

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
            </div>

            {/* Right: Modality contributions */}
            <div
              className="p-5 rounded-lg space-y-4"
              style={{ background: 'var(--bg-inset)', border: '1px solid var(--border-dim)' }}
            >
              <h4 className="text-[10px] font-bold uppercase tracking-widest" style={{ color: 'var(--text-3)' }}>
                Modality Contributions
              </h4>

              {results.data?.modality_scores && Object.entries(results.data.modality_scores).map(([name, score]) => (
                <ScoreBar key={name} name={name.toUpperCase()} score={score} />
              ))}

              <div className="pt-4" style={{ borderTop: '1px solid var(--border-dim)' }}>
                <p className="text-[10px] font-mono whitespace-pre-wrap leading-relaxed" style={{ color: 'var(--text-2)' }}>
                  {results.data?.explanation || JSON.stringify(results.data, null, 2)}
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
