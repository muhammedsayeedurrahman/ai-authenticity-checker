import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Film, Play, Settings2, ShieldCheck } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import RiskGauge from '../components/RiskGauge';
import VerdictCard from '../components/VerdictCard';
import FrameTable from '../components/FrameTable';
import useForensicStore from '../store/useForensicStore';

const fadeUp = {
  hidden:  { opacity: 0, y: 14 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.22, 1, 0.36, 1] } },
};

export default function VideoAnalysis() {
  const [file, setFile] = useState(null);
  const [fps, setFps] = useState(2);
  const [aggregation, setAggregation] = useState('weighted_avg');
  const { videoAnalysis, runVideoAnalysis } = useForensicStore();
  const { isAnalyzing, results, error } = videoAnalysis;

  const handleAnalyze = () => { if (file) runVideoAnalysis(file, fps, aggregation); };

  const videoUrl = results?.gradcam_video
    ? `data:video/mp4;base64,${results.gradcam_video}`
    : null;

  return (
    <motion.div initial="hidden" animate="visible" variants={fadeUp} className="space-y-5">
      {/* Header */}
      <header>
        <div className="flex items-center gap-2.5 mb-1.5">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center"
            style={{ background: 'var(--accent-dim)', border: '1px solid rgba(59,130,246,0.18)', boxShadow: '0 0 12px rgba(59,130,246,0.15)' }}
          >
            <Film size={14} style={{ color: 'var(--accent)' }} />
          </div>
          <h1 className="font-display text-xl font-bold" style={{ color: 'var(--text-1)' }}>Video Forensics</h1>
        </div>
        <p className="text-[12px]" style={{ color: 'var(--text-2)' }}>Frame-by-frame deepfake analysis with temporal consistency checking.</p>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-5">
        {/* Left panel */}
        <div className="xl:col-span-1 space-y-4">
          <UploadZone onFileSelect={setFile} accept="video/*" label="Drop video or click to browse" />

          {/* Parameters */}
          <div className="card">
            <div className="flex items-center gap-2 mb-3">
              <Settings2 size={13} style={{ color: 'var(--text-3)' }} />
              <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: 'var(--text-3)' }}>Parameters</span>
            </div>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-[11px]" style={{ color: 'var(--text-2)' }}>Sampling FPS</span>
                  <span className="text-[11px] font-bold font-mono" style={{ color: 'var(--accent)' }}>{fps} fps</span>
                </div>
                <input
                  type="range" min="1" max="15" step="0.5"
                  value={fps}
                  onChange={(e) => setFps(Number(e.target.value))}
                  className="w-full h-1 rounded-full cursor-pointer appearance-none"
                  style={{ accentColor: 'var(--accent)', background: 'var(--border-mid)' }}
                />
              </div>

              <div>
                <label className="text-[11px] mb-1.5 block" style={{ color: 'var(--text-2)' }}>Temporal Aggregation</label>
                <select
                  value={aggregation}
                  onChange={(e) => setAggregation(e.target.value)}
                  className="field-input text-[12px]"
                >
                  <option value="weighted_avg">Attention Weighted Avg</option>
                  <option value="max">Max Peak Risk</option>
                  <option value="average">Simple Average</option>
                  <option value="majority">Majority Vote</option>
                </select>
              </div>
            </div>
          </div>

          <button
            onClick={handleAnalyze}
            disabled={!file || isAnalyzing}
            className="btn-primary w-full py-3 text-[12px]"
          >
            {isAnalyzing ? (
              <>
                <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Processing Frames...
              </>
            ) : (
              <><Play size={15} /> RUN ANALYSIS</>
            )}
          </button>
        </div>

        {/* Right panel */}
        <div className="xl:col-span-3 space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Video preview */}
            <div className="card lg:col-span-2 flex items-center justify-center min-h-[300px] overflow-hidden">
              {videoUrl ? (
                <video src={videoUrl} controls autoPlay loop className="max-w-full max-h-full rounded-lg" />
              ) : (
                <div className="flex flex-col items-center justify-center space-y-2 p-6" style={{ color: 'var(--text-3)' }}>
                  <div className="w-12 h-12 rounded-xl flex items-center justify-center"
                    style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid var(--border-dim)' }}>
                    <Film size={20} className="opacity-20" />
                  </div>
                  <p className="text-[11px]">GradCAM Heatmap Video</p>
                </div>
              )}
            </div>

            {/* Results */}
            <div className="card lg:col-span-1 flex flex-col justify-center">
              {error ? (
                <div role="alert" className="p-3 rounded-lg text-[11px]"
                  style={{ background: 'rgba(251,113,133,0.08)', border: '1px solid rgba(251,113,133,0.20)', color: 'var(--risk-critical)' }}>
                  {error}
                </div>
              ) : results ? (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center">
                  <RiskGauge
                    percentage={results.risk_percentage || results.data?.risk_percent || 0}
                    label="Video Avg Risk"
                    size={170}
                  />
                  <div className="w-full mt-1">
                    <VerdictCard verdict={results.verdict || results.data?.verdict} />
                  </div>
                </motion.div>
              ) : (
                <div className="flex flex-col items-center justify-center space-y-2 py-8" style={{ color: 'var(--text-3)' }}>
                  <ShieldCheck size={20} className="opacity-20" />
                  <p className="text-[11px]">Awaiting results</p>
                </div>
              )}
            </div>
          </div>

          {/* Frame table */}
          {results?.frame_details && (
            <div className="card">
              <p className="text-[10px] font-bold uppercase tracking-widest mb-3" style={{ color: 'var(--text-3)' }}>Frame Timeline</p>
              <FrameTable framesRawStr={results.frame_details} />
            </div>
          )}

          {/* Details */}
          {(results?.details || results?.data?.explanation) && (
            <div className="card">
              <p className="text-[10px] font-bold uppercase tracking-widest mb-2" style={{ color: 'var(--text-3)' }}>Intelligence Details</p>
              <pre
                className="text-[10px] font-mono whitespace-pre-wrap leading-relaxed p-3 rounded-lg"
                style={{ background: 'var(--bg-inset)', border: '1px solid var(--border-dim)', color: 'var(--text-2)' }}
              >
                {results.details || JSON.stringify(results.data?.temporal_analysis, null, 2) || results.data?.explanation}
              </pre>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
