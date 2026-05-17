import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { ScanSearch, ShieldCheck, Cpu, AlertTriangle } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import RiskGauge from '../components/RiskGauge';
import ScoreBar from '../components/ScoreBar';
import VerdictCard from '../components/VerdictCard';
import HeatmapViewer from '../components/HeatmapViewer';
import useForensicStore from '../store/useForensicStore';

const fadeUp = {
  hidden:  { opacity: 0, y: 14 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.22, 1, 0.36, 1] } },
};

export default function ImageAnalysis() {
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState('Full Ensemble (7 models)');
  const { systemStatus, imageAnalysis, runImageAnalysis } = useForensicStore();
  const { isAnalyzing, results, error } = imageAnalysis;
  const fastModeAvailable = systemStatus?.corefakenet_available;

  const handleAnalyze = () => { if (file) runImageAnalysis(file, mode); };

  return (
    <motion.div initial="hidden" animate="visible" variants={fadeUp} className="space-y-6">

      {/* Header */}
      <header className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2.5 mb-1.5">
            <div
              className="w-7 h-7 rounded-lg flex items-center justify-center"
              style={{ background: 'var(--accent-dim)', border: '1px solid rgba(59,130,246,0.18)', boxShadow: '0 0 12px rgba(59,130,246,0.15)' }}
            >
              <ScanSearch size={14} style={{ color: 'var(--accent)' }} />
            </div>
            <h1 className="font-display text-xl font-bold" style={{ color: 'var(--text-1)' }}>Image Analysis</h1>
          </div>
          <p className="text-[12px]" style={{ color: 'var(--text-2)' }}>
            Detect manipulation, AI generation, and face-swapping in still images.
          </p>
        </div>
      </header>

      {/* Main grid */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-5">

        {/* Left panel */}
        <div className="lg:col-span-1 space-y-4">

          {/* Mode selector */}
          <div className="card">
            <div className="flex items-center gap-2 mb-3">
              <Cpu size={13} style={{ color: 'var(--text-3)' }} />
              <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: 'var(--text-3)' }}>Analysis Mode</span>
            </div>

            <div className="space-y-2">
              {[
                { value: 'Full Ensemble (7 models)', label: 'Full Ensemble', sub: '7 models · High accuracy', available: true },
                { value: 'Fast Mode (CorefakeNet)',  label: 'Fast Mode',     sub: 'Single-pass · CorefakeNet', available: fastModeAvailable },
              ].map((opt) => (
                <label
                  key={opt.value}
                  className={`flex items-center gap-3 p-2.5 rounded-lg cursor-pointer transition-all duration-200 ${
                    !opt.available ? 'opacity-40 cursor-not-allowed' : ''
                  }`}
                  style={{
                    border: mode === opt.value
                      ? '1px solid rgba(59,130,246,0.30)'
                      : '1px solid var(--border-dim)',
                    background: mode === opt.value
                      ? 'var(--accent-dim)'
                      : 'rgba(255,255,255,0.02)',
                  }}
                >
                  <span
                    className="flex-shrink-0 w-3.5 h-3.5 rounded-full flex items-center justify-center"
                    style={{
                      border: mode === opt.value
                        ? '4px solid var(--accent)'
                        : '1.5px solid rgba(255,255,255,0.2)',
                      transition: 'border 0.2s ease',
                    }}
                  />
                  <input
                    type="radio"
                    name="mode"
                    value={opt.value}
                    checked={mode === opt.value}
                    onChange={(e) => opt.available && setMode(e.target.value)}
                    disabled={!opt.available}
                    className="sr-only"
                  />
                  <div>
                    <p className="text-[12px] font-semibold leading-none" style={{ color: 'var(--text-1)' }}>{opt.label}</p>
                    <p className="text-[10px] mt-0.5" style={{ color: 'var(--text-2)' }}>{opt.sub}</p>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Upload */}
          <UploadZone onFileSelect={setFile} label="Drop image or click to browse" />

          {/* Analyze button */}
          <button
            onClick={handleAnalyze}
            disabled={!file || isAnalyzing}
            className="btn-primary w-full py-3 text-[12px]"
          >
            {isAnalyzing ? (
              <>
                <span
                  className="w-3.5 h-3.5 border-2 border-current border-t-transparent rounded-full animate-spin"
                  style={{ borderColor: '#FFFFFF', borderTopColor: 'transparent' }}
                />
                Analyzing...
              </>
            ) : (
              <>
                <ShieldCheck size={15} />
                RUN FORENSIC ANALYSIS
              </>
            )}
          </button>
        </div>

        {/* Heatmap */}
        <div className="lg:col-span-2">
          <HeatmapViewer
            originalFile={file}
            gradcamBase64={results?.gradcam || results?.data?.gradcam_image}
          />
        </div>

        {/* Report panel */}
        <div className="lg:col-span-1">
          <div className="card min-h-[420px]">
            <div className="flex items-center gap-2 mb-3">
              <AlertTriangle size={13} style={{ color: 'var(--text-3)' }} />
              <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: 'var(--text-3)' }}>Intelligence Report</span>
            </div>

            {error ? (
              <div
                role="alert"
                className="p-3 rounded-lg text-[11px] mt-4"
                style={{ background: 'rgba(251,113,133,0.08)', border: '1px solid rgba(251,113,133,0.20)', color: 'var(--risk-critical)' }}
              >
                {error}
              </div>
            ) : !results ? (
              <div className="h-full flex flex-col items-center justify-center mt-20 space-y-2" style={{ color: 'var(--text-3)' }}>
                <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-dim)' }}>
                  <ShieldCheck size={18} className="opacity-30" />
                </div>
                <p className="text-[11px]">Awaiting payload</p>
              </div>
            ) : (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
                <RiskGauge percentage={results.risk_percentage || results.data?.risk_percent || 0} />

                {results.risk_label && (
                  <div
                    className="text-center py-1.5 rounded-lg text-[11px] font-semibold tracking-wide"
                    style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid var(--border-dim)', color: 'var(--text-2)' }}
                  >
                    {results.risk_label}
                  </div>
                )}

                <div className="mt-3 space-y-1">
                  {results.model_scores && Object.entries(results.model_scores).map(([name, score]) => (
                    <ScoreBar key={name} name={name} score={score} />
                  ))}
                  {results.data?.model_scores && Object.entries(results.data.model_scores).map(([name, score]) => (
                    <ScoreBar key={name} name={name} score={score} />
                  ))}
                </div>

                <VerdictCard verdict={results.verdict || results.data?.verdict} />

                {(results.details || results.data?.explanation) && (
                  <div
                    className="p-3 rounded-lg mt-2"
                    style={{ background: 'var(--bg-inset)', border: '1px solid var(--border-dim)' }}
                  >
                    <p className="text-[10px] font-mono whitespace-pre-wrap leading-relaxed" style={{ color: 'var(--text-2)' }}>
                      {results.details || results.data.explanation}
                    </p>
                  </div>
                )}
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
