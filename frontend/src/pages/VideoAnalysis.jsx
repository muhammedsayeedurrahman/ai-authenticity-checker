import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Film, Play, Settings2, ShieldCheck, X, Zap } from 'lucide-react';
import PageHeader from '../components/PageHeader';
import ConfirmDialog from '../components/ConfirmDialog';
import UploadZone from '../components/UploadZone';
import RiskGauge from '../components/RiskGauge';
import VerdictCard from '../components/VerdictCard';
import FrameTable from '../components/FrameTable';
import VideoRiskTimeline from '../components/VideoRiskTimeline';
import IndeterminateProgress from '../components/IndeterminateProgress';
import useForensicStore from '../store/useForensicStore';

const MODES = [
  {
    value: 'ensemble',
    label: 'Full Ensemble',
    sub: '7 models/frame -- Most thorough, slow on CPU',
  },
  {
    value: 'fast',
    label: 'Fast Mode',
    sub: 'CorefakeNet single-pass -- Seconds, not minutes',
  },
];

export default function VideoAnalysis() {
  const [file, setFile] = useState(null);
  const [fps, setFps] = useState(1);
  const [aggregation, setAggregation] = useState('weighted_avg');
  const [mode, setMode] = useState(MODES[0].value);
  const [videoUrl, setVideoUrl] = useState(null);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const blobUrlRef = useRef(null);
  const {
    videoAnalysis, runVideoAnalysis, clearAnalysis, pendingFile, clearPendingFile, systemStatus,
  } = useForensicStore();
  const { isAnalyzing, results, error } = videoAnalysis;
  const fastModeAvailable = systemStatus?.corefakenet_available;

  useEffect(() => {
    if (pendingFile) {
      setFile(pendingFile);
      clearPendingFile();
    }
  }, [pendingFile, clearPendingFile]);

  useEffect(() => {
    if (blobUrlRef.current) {
      URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
    }
    if (file) {
      const url = URL.createObjectURL(file);
      blobUrlRef.current = url;
      setVideoUrl(url);
    } else {
      setVideoUrl(null);
    }
    return () => {
      if (blobUrlRef.current) {
        URL.revokeObjectURL(blobUrlRef.current);
        blobUrlRef.current = null;
      }
    };
  }, [file]);

  const handleFileSelect = useCallback((f) => setFile(f), []);
  const handleAnalyze = () => { if (file) runVideoAnalysis(file, fps, aggregation, mode); };
  const handleCancelConfirm = useCallback(() => {
    const { cancelAnalysis } = useForensicStore.getState();
    cancelAnalysis('video');
    clearAnalysis('video');
    setConfirmCancel(false);
  }, [clearAnalysis]);

  return (
    <div className="space-y-5">
      <PageHeader
        icon={Film}
        title="Video Forensics"
        subtitle="Frame-by-frame deepfake analysis with temporal consistency checking."
      />

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-5">
        {/* Left panel */}
        <div className="lg:col-span-1 space-y-4">
          <UploadZone onFileSelect={handleFileSelect} accept="video/*" label="Drop video or click to browse" />

          {/* Analysis mode */}
          <div className="card">
            <div className="flex items-center gap-2 mb-3">
              <Zap size={13} className="text-text-3" />
              <span className="label-tag">Analysis Mode</span>
            </div>

            <div className="space-y-2">
              {MODES.map((opt) => {
                const available = opt.value === MODES[0].value || fastModeAvailable;
                const selected = mode === opt.value;

                return (
                  <label
                    key={opt.value}
                    className={`flex items-center gap-3 p-2.5 rounded-lg cursor-pointer transition-all duration-200 ${
                      !available ? 'opacity-40 cursor-not-allowed' : ''
                    } ${selected ? 'bg-accent-dim border border-accent/30' : 'border border-border-dim bg-white/[0.02]'}`}
                  >
                    <span
                      className={`flex-shrink-0 w-3.5 h-3.5 rounded-full transition-[border] duration-200 ${
                        selected
                          ? 'border-[4px] border-accent'
                          : 'border-[1.5px] border-white/20'
                      }`}
                    />
                    <input
                      type="radio"
                      name="video-analysis-mode"
                      value={opt.value}
                      checked={selected}
                      onChange={(e) => available && setMode(e.target.value)}
                      disabled={!available}
                      className="sr-only"
                    />
                    <div>
                      <p className="text-sm font-semibold leading-none text-text-1">
                        {opt.label}
                      </p>
                      <p className="text-xs mt-0.5 text-text-2">
                        {opt.sub}
                      </p>
                    </div>
                  </label>
                );
              })}
            </div>
          </div>

          {/* Parameters */}
          <div className="card">
            <div className="flex items-center gap-2 mb-3">
              <Settings2 size={14} className="text-text-3" />
              <span className="label-tag">Parameters</span>
            </div>

            <div className="space-y-4">
              <div>
                <div className="flex justify-between mb-2">
                  <span className="text-xs text-text-2">Sampling FPS</span>
                  <span className="text-xs font-bold font-mono text-accent">{fps} fps</span>
                </div>
                <input
                  type="range" min="1" max="15" step="0.5"
                  value={fps}
                  onChange={(e) => setFps(Number(e.target.value))}
                  className="w-full h-1 rounded-full cursor-pointer appearance-none bg-border-mid [accent-color:var(--accent)]"
                />
              </div>

              <div>
                <label className="text-xs mb-1.5 block text-text-2">
                  Temporal Aggregation
                </label>
                <select
                  value={aggregation}
                  onChange={(e) => setAggregation(e.target.value)}
                  className="field-input text-sm"
                >
                  <option value="weighted_avg">Attention Weighted Avg</option>
                  <option value="max">Max Peak Risk</option>
                  <option value="average">Simple Average</option>
                  <option value="majority">Majority Vote</option>
                </select>
              </div>
            </div>
          </div>

          <div className="flex gap-2">
            <button
              onClick={handleAnalyze}
              disabled={!file || isAnalyzing}
              className="btn-primary flex-1 py-3"
            >
              {isAnalyzing ? (
                <>
                  <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Processing...
                </>
              ) : (
                <><Play size={15} /> Analyze</>
              )}
            </button>
            {isAnalyzing && (
              <button onClick={() => setConfirmCancel(true)} className="btn-danger px-3" title="Cancel">
                <X size={15} />
              </button>
            )}
          </div>
          {isAnalyzing && (
            <IndeterminateProgress label="Extracting & scoring frames…" />
          )}
        </div>

        {/* Right panel */}
        <div className="lg:col-span-3 space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {/* Video preview */}
            <div className="card lg:col-span-2 flex items-center justify-center min-h-[300px] overflow-hidden">
              {videoUrl ? (
                <video
                  src={videoUrl}
                  controls
                  className="w-full max-h-[400px] rounded-lg object-contain bg-bg-inset"
                >
                  Your browser does not support the video tag.
                </video>
              ) : (
                <div className="flex flex-col items-center space-y-2 text-text-3">
                  <Film size={22} className="opacity-30" />
                  <p className="text-sm">Upload a video to preview</p>
                </div>
              )}
            </div>

            {/* Results */}
            <div className="card lg:col-span-1 flex flex-col justify-center">
              {error ? (
                <div
                  role="alert"
                  className="p-3 rounded-lg text-sm bg-risk-criticalDim text-risk-critical border border-[rgba(251,113,133,0.20)]"
                >
                  {error}
                </div>
              ) : results ? (
                <div className="flex flex-col items-center">
                  <RiskGauge
                    percentage={results.risk_percent || 0}
                    label="Video Avg Risk"
                    size={170}
                  />
                  <div className="w-full mt-1">
                    <VerdictCard verdict={results.verdict} riskScore={results.risk_percent} />
                  </div>

                  {results.total_frames_analyzed > 0 && (
                    <div className="w-full mt-3 grid grid-cols-3 gap-2 text-center">
                      {[
                        { label: 'Frames', value: results.total_frames_analyzed },
                        { label: 'Fake', value: results.fake_frames || 0 },
                        { label: 'Real', value: results.real_frames || 0 },
                      ].map(({ label, value }) => (
                        <div key={label} className="inset-panel py-2">
                          <p className="text-base font-bold font-mono text-text-1">{value}</p>
                          <p className="text-xs text-text-3">{label}</p>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center space-y-2 py-8 text-text-3">
                  <ShieldCheck size={22} className="opacity-30" />
                  <p className="text-sm">Awaiting results</p>
                </div>
              )}
            </div>
          </div>

          {results?.frame_details && (
            <>
              <VideoRiskTimeline framesRawStr={results.frame_details} />

              <div className="card">
                <p className="label-tag mb-3">Frame Detail</p>
                <FrameTable framesRawStr={results.frame_details} />
              </div>
            </>
          )}

          {results?.explanation && (
            <div className="card">
              <p className="label-tag mb-2">Analysis Details</p>
              <p className="text-sm leading-relaxed text-text-2">
                {results.explanation}
              </p>
            </div>
          )}
        </div>
      </div>

      <ConfirmDialog
        open={confirmCancel}
        title="Cancel Analysis"
        message="The video analysis is still processing frames. Are you sure you want to cancel?"
        confirmLabel="Cancel Analysis"
        onConfirm={handleCancelConfirm}
        onCancel={() => setConfirmCancel(false)}
      />
    </div>
  );
}
