import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  Film,
  Play,
  Settings2,
  ShieldCheck,
  ShieldAlert,
  X,
  Zap,
  Layers,
  Activity,
  Sliders,
  Sparkles,
  FolderOpen,
  RefreshCw,
  Clock,
  Video as VideoIcon,
  BarChart2,
  Info,
} from 'lucide-react';
import { fadeUp } from '../utils/animations';
import PageHeader from '../components/PageHeader';
import ConfirmDialog from '../components/ConfirmDialog';
import ComplaintModal from '../components/ComplaintModal';
import RiskGauge from '../components/RiskGauge';
import CybercrimeRiskAdvisory from '../components/CybercrimeRiskAdvisory';
import ComplianceLabelBadge from '../components/ComplianceLabelBadge';
import SnakeLoader from '../components/SnakeLoader';
import FrameTable from '../components/FrameTable';
import VideoRiskTimeline from '../components/VideoRiskTimeline';
import useForensicStore from '../store/useForensicStore';
import { isFileAccepted } from '../utils/format';
import { getRiskColorRaw, getRiskLevel } from '../utils/risk';

const MODES = [
  {
    value: 'ensemble',
    label: 'Full Ensemble',
    tag: 'DEEP FORENSIC',
    sub: '7 models/frame • ViT, FFT, DINOv2 & Facial Biometrics',
    icon: Layers,
  },
  {
    value: 'fast',
    label: 'Fast CorefakeNet',
    tag: 'LOWER ACCURACY',
    sub: 'Single-pass neural network • ~76% validation accuracy — quick screening only, not a final verdict',
    icon: Zap,
    warn: true,
  },
];

// Maps the backend's `fusion_mode` (what actually scored this result) back to
// the mode label above — used so results/history always report the model
// that really ran, not just whatever the selector happens to be set to.
const FUSION_MODE_LABEL = {
  video_ensemble_7model: 'Full Ensemble',
  corefakenet_fast: 'Fast CorefakeNet',
};

const VIDEO_SCAN_STEPS = [
  'Decoding video stream & extracting keyframes...',
  'Running facial alignment & bounding box detection...',
  'Evaluating spatial & temporal consistency across frames...',
  'Computing temporal attention-weighted risk aggregation...',
];

export default function VideoAnalysis() {
  const [file, setFile] = useState(null);
  const fps = 1;
  const aggregation = 'weighted_avg';
  const [mode, setMode] = useState(MODES[0].value);
  const [videoUrl, setVideoUrl] = useState(null);
  const [fileMeta, setFileMeta] = useState(null);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [scanStepIndex, setScanStepIndex] = useState(0);
  const [complaintOpen, setComplaintOpen] = useState(false);

  const fileInputRef = useRef(null);
  const videoRef = useRef(null);
  const blobUrlRef = useRef(null);

  const {
    videoAnalysis,
    runVideoAnalysis,
    clearAnalysis,
    pendingFile,
    clearPendingFile,
    systemStatus,
  } = useForensicStore();

  const { isAnalyzing, results, error } = videoAnalysis;
  const fastModeAvailable = systemStatus?.corefakenet_available;

  const handleLoadFile = useCallback((selectedFile) => {
    if (!selectedFile) {
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
      setFile(null);
      setVideoUrl(null);
      setFileMeta(null);
      return;
    }

    if (!isFileAccepted(selectedFile, 'video/*')) return;

    if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
    const url = URL.createObjectURL(selectedFile);
    blobUrlRef.current = url;

    setFile(selectedFile);
    setVideoUrl(url);
    setFileMeta({
      name: selectedFile.name,
      size: (selectedFile.size / (1024 * 1024)).toFixed(2) + ' MB',
      type: selectedFile.type || 'video/mp4',
    });
  }, []);

  useEffect(() => {
    if (pendingFile) {
      handleLoadFile(pendingFile);
      clearPendingFile();
    }
  }, [pendingFile, clearPendingFile, handleLoadFile]);

  useEffect(() => {
    return () => {
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
    };
  }, []);

  useEffect(() => {
    if (!isAnalyzing) {
      setScanStepIndex(0);
      return;
    }
    const interval = setInterval(() => {
      setScanStepIndex((prev) => (prev + 1) % VIDEO_SCAN_STEPS.length);
    }, 2200);
    return () => clearInterval(interval);
  }, [isAnalyzing]);

  const handleAnalyze = () => {
    if (file && !isAnalyzing) {
      runVideoAnalysis(file, fps, aggregation, mode);
    }
  };

  const handleCancelConfirm = useCallback(() => {
    const { cancelAnalysis } = useForensicStore.getState();
    cancelAnalysis('video');
    clearAnalysis('video');
    setConfirmCancel(false);
  }, [clearAnalysis]);

  return (
    <motion.div initial="hidden" animate="visible" variants={fadeUp} className="space-y-6 pb-12">
      <PageHeader
        icon={Film}
        title="Video & Deepfake Forensics"
        subtitle="Frame-by-frame deepfake inspection, temporal continuity validation, and facial manipulation analysis."
      />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Video Player & Parameters (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          {/* Video Player Card */}
          <div className="card space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-purple-600 animate-pulse" />
                <span className="label-tag">Video Monitor</span>
              </div>
              {fileMeta && (
                <span className="text-[11px] font-mono px-2.5 py-0.5 rounded-full bg-purple-100 text-purple-900 font-semibold border border-purple-200">
                  {fileMeta.size} • {fileMeta.type.replace('video/', '').toUpperCase()}
                </span>
              )}
            </div>

            <div
              onClick={() => { if (!videoUrl) fileInputRef.current?.click(); }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const dropped = e.dataTransfer.files[0];
                if (dropped) handleLoadFile(dropped);
              }}
              className={`relative w-full rounded-2xl flex flex-col items-center justify-center min-h-[260px] max-h-[380px] overflow-hidden border-2 border-dashed transition-all duration-300 ${
                videoUrl
                  ? 'border-purple-300 bg-black/80 shadow-inner'
                  : 'border-purple-300/80 hover:border-purple-500 bg-purple-50/50 cursor-pointer hover:bg-purple-100/40'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={(e) => {
                  if (e.target.files?.[0]) handleLoadFile(e.target.files[0]);
                  e.target.value = '';
                }}
                className="hidden"
              />

              {videoUrl ? (
                <div className="relative w-full h-[280px] flex items-center justify-center bg-black/90">
                  {isAnalyzing && <div className="scan-beam" />}

                  <video
                    ref={videoRef}
                    src={videoUrl}
                    controls
                    className="w-full h-full object-contain rounded-xl"
                  >
                    Your browser does not support the video tag.
                  </video>

                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleLoadFile(null);
                      clearAnalysis('video');
                    }}
                    className="absolute top-3 right-3 z-20 p-1.5 rounded-full bg-black/70 hover:bg-black text-white hover:text-rose-400 transition-all"
                    title="Remove video"
                  >
                    <X size={14} />
                  </button>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center p-6 text-center">
                  <div className="w-16 h-16 rounded-3xl bg-white border border-purple-200 shadow-md flex items-center justify-center mb-3 text-purple-600 hover:scale-105 transition-transform">
                    <Film size={30} />
                  </div>
                  <p className="text-sm font-bold text-[#1E1238] mb-1">
                    Upload Video Target
                  </p>
                  <p className="text-xs text-[#5B4E75] max-w-[240px] mb-4">
                    Supports MP4, MOV, AVI, WEBM for temporal deepfake frame evaluation
                  </p>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      fileInputRef.current?.click();
                    }}
                    className="btn-ghost py-1.5 px-4 text-xs font-bold"
                  >
                    <FolderOpen size={13} /> Browse Video Files
                  </button>
                </div>
              )}
            </div>

            {videoUrl && (
              <div className="flex gap-2 pt-1">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="btn-ghost flex-1 py-2 text-xs font-bold"
                >
                  <RefreshCw size={12} /> Replace Video
                </button>
                <button
                  onClick={() => {
                    handleLoadFile(null);
                    clearAnalysis('video');
                  }}
                  className="btn-danger py-2 px-4 text-xs font-bold"
                >
                  <X size={12} /> Clear
                </button>
              </div>
            )}
          </div>

          {/* Forensic Parameters Card */}
          <div className="card space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Settings2 size={15} className="text-purple-700" />
                <span className="label-tag">Temporal Parameters</span>
              </div>
              <span className="text-[10px] font-mono font-bold text-purple-700 bg-purple-100 px-2 py-0.5 rounded-full">
                TEMPORAL v2.3
              </span>
            </div>

            {/* Mode Selector */}
            <div className="space-y-2.5">
              {MODES.map((opt) => {
                const available = opt.value === MODES[0].value || fastModeAvailable;
                const selected = mode === opt.value;
                const Icon = opt.icon;

                return (
                  <label
                    key={opt.value}
                    className={`relative flex items-start gap-3.5 p-3.5 rounded-2xl cursor-pointer transition-all duration-200 border ${
                      !available ? 'opacity-40 cursor-not-allowed' : ''
                    } ${
                      selected
                        ? 'bg-purple-100/60 border-purple-400 shadow-sm shadow-purple-500/10'
                        : 'border-purple-200/60 bg-white/70 hover:border-purple-300'
                    }`}
                  >
                    <input
                      type="radio"
                      name="video-mode"
                      value={opt.value}
                      checked={selected}
                      onChange={(e) => available && setMode(e.target.value)}
                      disabled={!available}
                      className="sr-only"
                    />
                    <div
                      className={`w-4 h-4 mt-0.5 rounded-full flex items-center justify-center transition-all ${
                        selected
                          ? 'border-[4px] border-purple-700 bg-white'
                          : 'border-[1.5px] border-purple-300 bg-transparent'
                      }`}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between mb-0.5">
                        <span className="text-xs sm:text-sm font-bold text-[#1E1238] flex items-center gap-1.5">
                          <Icon size={14} className={selected ? 'text-purple-700' : 'text-purple-400'} />
                          {opt.label}
                        </span>
                        <span
                          className={`text-[9px] font-mono px-2 py-0.5 rounded-full font-bold tracking-wider ${
                            opt.warn
                              ? selected
                                ? 'bg-amber-500 text-white'
                                : 'bg-amber-100 text-amber-800'
                              : selected
                                ? 'bg-purple-700 text-white'
                                : 'bg-purple-100 text-purple-800'
                          }`}
                        >
                          {opt.tag}
                        </span>
                      </div>
                      <p className="text-xs text-[#5B4E75] leading-relaxed">{opt.sub}</p>
                    </div>
                  </label>
                );
              })}
            </div>

            <p className="flex items-start gap-1.5 text-[11px] leading-relaxed text-[#8F81A8]">
              <Info size={12} className="mt-0.5 flex-shrink-0 text-purple-400" />
              Each mode runs a different model — the same video can score differently
              across modes. This reflects methodology, not an unreliable score.
            </p>

            {/* Run CTA */}
            {isAnalyzing ? (
              <div className="space-y-3 pt-1">
                <div className="flex gap-2">
                  <button disabled className="btn-primary flex-1 py-3 text-xs sm:text-sm font-bold">
                    <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Extracting & Scoring Frames...
                  </button>
                  <button onClick={() => setConfirmCancel(true)} className="btn-danger py-3 px-4 text-xs font-bold">
                    <X size={15} /> Cancel
                  </button>
                </div>
                <div className="p-3.5 rounded-2xl bg-white border border-purple-200 space-y-2 shadow-sm">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-purple-700 flex items-center gap-1.5 font-bold">
                      <Sparkles size={13} className="animate-spin text-purple-600" />
                      {VIDEO_SCAN_STEPS[scanStepIndex]}
                    </span>
                    <span className="text-[10px] font-mono text-purple-500 font-bold">STEP {scanStepIndex + 1}/4</span>
                  </div>
                  <div className="progress-indeterminate-track" />
                </div>
              </div>
            ) : (
              <button
                onClick={handleAnalyze}
                disabled={!file}
                className="btn-primary w-full py-3.5 text-sm font-bold shadow-md shadow-purple-900/20"
              >
                <Play size={18} />
                Initiate Video Forensics
              </button>
            )}
          </div>
        </div>

        {/* Right Column: Temporal Forensics Station (7 cols) */}
        <div className="lg:col-span-7 space-y-5">
          {error ? (
            <div className="card p-6 border-rose-200 bg-rose-50/70 space-y-3">
              <div className="flex items-center gap-2 text-rose-700 font-bold">
                <ShieldAlert size={20} />
                <h3>Video Forensic Error</h3>
              </div>
              <p className="text-xs sm:text-sm text-rose-900 leading-relaxed">{error}</p>
              <button onClick={handleAnalyze} className="btn-primary py-2 px-4 text-xs font-bold mt-2">
                Retry Scan
              </button>
            </div>
          ) : !results ? (
            /* Futuristic Video Station Empty State */
            <div className="card p-8 min-h-[480px] flex flex-col items-center justify-center text-center relative overflow-hidden">
              <div className="max-w-md space-y-5 relative z-10">
                <div className="flex items-center justify-center">
                  <SnakeLoader
                    width={9}
                    speed={90}
                    playing={isAnalyzing}
                    snakeColor="#6D28D9"
                    appleColor="#EC4899"
                    className="gap-[3px]"
                    dotClassName="size-2 rounded-[2px]"
                  />
                </div>

                <div>
                  <h3 className="text-lg font-black text-[#1E1238]">
                    {isAnalyzing ? 'Decoding Temporal Signal…' : 'Temporal Deepfake Station Ready'}
                  </h3>
                  <p className="text-xs sm:text-sm text-[#5B4E75] mt-1.5 leading-relaxed">
                    Upload a video to decode frame streams, calculate temporal consistency curves, and evaluate face-swapping manipulation.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3 pt-2 text-left">
                  <div className="p-3.5 rounded-2xl bg-white border border-purple-100 shadow-sm">
                    <div className="text-xs font-bold text-purple-700 mb-1 flex items-center gap-1">
                      <BarChart2 size={13} /> Temporal Risk Curve
                    </div>
                    <p className="text-[11px] text-[#5B4E75]">
                      Visual second-by-second manipulation probability trajectory.
                    </p>
                  </div>
                  <div className="p-3.5 rounded-2xl bg-white border border-purple-100 shadow-sm">
                    <div className="text-xs font-bold text-cyan-600 mb-1 flex items-center gap-1">
                      <Activity size={13} /> Frame Matrix
                    </div>
                    <p className="text-[11px] text-[#5B4E75]">
                      Detailed breakdown of every extracted frame with model scores.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Results Hub */
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-5">
              {/* Verdict Banner */}
              <div
                className={`card p-6 border relative overflow-hidden ${
                  results.risk_percent > 50
                    ? 'border-rose-300 bg-gradient-to-br from-rose-50 via-white to-rose-50/40 shadow-lg shadow-rose-500/10'
                    : 'border-emerald-300 bg-gradient-to-br from-emerald-50 via-white to-emerald-50/40 shadow-lg shadow-emerald-500/10'
                }`}
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="label-tag text-[10px]">Video Forensic Classification</span>
                    <span
                      className="text-[10px] font-mono px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider"
                      style={{
                        color: getRiskColorRaw(results.risk_percent || 0),
                        backgroundColor: `${getRiskColorRaw(results.risk_percent || 0)}15`,
                        border: `1px solid ${getRiskColorRaw(results.risk_percent || 0)}30`,
                      }}
                    >
                      {getRiskLevel(results.risk_percent || 0)}
                    </span>
                    {results.fusion_mode && (
                      <span
                        className={`text-[10px] font-mono px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider border ${
                          results.fusion_mode === 'corefakenet_fast'
                            ? 'bg-amber-100 text-amber-800 border-amber-200'
                            : 'bg-purple-100 text-purple-700 border-purple-200'
                        }`}
                        title={
                          results.fusion_mode === 'corefakenet_fast'
                            ? 'Fast mode: single lightweight model, ~76% validation accuracy — treat as a quick screen, not a final verdict'
                            : 'The model pipeline that produced this score'
                        }
                      >
                        Scored via {FUSION_MODE_LABEL[results.fusion_mode] || results.fusion_mode}
                      </span>
                    )}
                  </div>
                  <h2 className="text-2xl sm:text-3xl font-black font-display tracking-tight text-[#1E1238] flex items-center gap-2.5">
                    {results.risk_percent > 50 ? (
                      <ShieldAlert className="text-rose-600 flex-shrink-0" size={30} />
                    ) : (
                      <ShieldCheck className="text-emerald-600 flex-shrink-0" size={30} />
                    )}
                    {results.verdict || 'Temporal Analysis Complete'}
                  </h2>
                  <p className="text-xs sm:text-sm text-[#5B4E75] leading-relaxed max-w-xl">
                    {results.explanation || 'Aggregated multi-frame risk score computed across all extracted video frames.'}
                  </p>
                </div>

                {results.verdict === 'AI-GENERATED' && (
                  <button
                    onClick={() => setComplaintOpen(true)}
                    className="btn-danger w-full py-2.5 text-xs sm:text-sm font-bold mt-4"
                  >
                    <ShieldAlert size={15} />
                    Raise Cyber Crime Complaint
                  </button>
                )}
              </div>

              {/* Compliance & cybercrime advisories (India IT Rules 2026) */}
              <CybercrimeRiskAdvisory risk={results.cybercrime_risk} />
              <ComplianceLabelBadge label={results.compliance_label} />

              {/* Gauge & Frame Metrics */}
              <div className="grid grid-cols-1 sm:grid-cols-12 gap-4">
                <div className="sm:col-span-5 card p-4 flex flex-col items-center justify-center text-center">
                  <RiskGauge
                    percentage={results.risk_percent || 0}
                    label="VIDEO RISK"
                    size={150}
                    strokeWidth={10}
                    showBadge={true}
                  />
                </div>

                <div className="sm:col-span-7 card p-5 flex flex-col justify-between space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="label-tag text-[10px]">Frame Telemetry</span>
                    <span className="text-[10px] font-mono font-bold text-purple-700 bg-purple-100 px-2 py-0.5 rounded-full">
                      {fps} FPS SAMPLING
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-2">
                    <div className="p-3 rounded-2xl bg-white border border-purple-100 text-center shadow-sm">
                      <p className="text-lg font-black font-mono text-[#1E1238]">
                        {results.total_frames_analyzed || 0}
                      </p>
                      <p className="text-[9px] text-[#8F81A8] font-bold uppercase tracking-wider mt-0.5">Analyzed</p>
                    </div>
                    <div className="p-3 rounded-2xl bg-rose-50 border border-rose-200 text-center">
                      <p className="text-lg font-black font-mono text-rose-600">
                        {results.fake_frames || 0}
                      </p>
                      <p className="text-[9px] text-rose-700 font-bold uppercase tracking-wider mt-0.5">Manipulated</p>
                    </div>
                    <div className="p-3 rounded-2xl bg-emerald-50 border border-emerald-200 text-center">
                      <p className="text-lg font-black font-mono text-emerald-600">
                        {results.real_frames || 0}
                      </p>
                      <p className="text-[9px] text-emerald-700 font-bold uppercase tracking-wider mt-0.5">Authentic</p>
                    </div>
                  </div>

                  <div className="text-xs text-[#8F81A8] flex justify-between pt-1 border-t border-purple-100 font-medium">
                    <span>Aggregation Logic:</span>
                    <span className="font-mono text-purple-900 font-bold capitalize">{aggregation.replace('_', ' ')}</span>
                  </div>
                </div>
              </div>

              {/* Video Risk Timeline */}
              {results.frame_details && (
                <div className="space-y-4">
                  <VideoRiskTimeline framesRawStr={results.frame_details} />

                  <div className="card p-5 space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Activity size={14} className="text-purple-700" />
                        <span className="label-tag">Frame-by-Frame Matrix</span>
                      </div>
                      <span className="text-xs font-mono font-bold text-purple-700 bg-purple-100 px-2.5 py-0.5 rounded-full">
                        {results.total_frames_analyzed || 0} Frames Logged
                      </span>
                    </div>
                    <FrameTable framesRawStr={results.frame_details} />
                  </div>
                </div>
              )}

              {/* Summary */}
              {results.explanation && (
                <div className="card p-5 space-y-2">
                  <div className="flex items-center gap-2">
                    <Sliders size={14} className="text-purple-700" />
                    <span className="label-tag">Temporal Forensics Log</span>
                  </div>
                  <p className="text-xs font-mono leading-relaxed text-[#5B4E75] p-3.5 rounded-xl bg-purple-50/70 border border-purple-100 whitespace-pre-wrap">
                    {results.explanation}
                  </p>
                </div>
              )}
            </motion.div>
          )}
        </div>
      </div>

      <ConfirmDialog
        open={confirmCancel}
        title="Cancel Video Forensics"
        message="Are you sure you want to stop the active frame analysis pipeline?"
        confirmLabel="Cancel Analysis"
        onConfirm={handleCancelConfirm}
        onCancel={() => setConfirmCancel(false)}
      />

      <ComplaintModal
        open={complaintOpen}
        onClose={() => setComplaintOpen(false)}
        analysis={results}
        fileName={file?.name}
      />
    </motion.div>
  );
}
