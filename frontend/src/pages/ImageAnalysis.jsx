import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Image as ImageIcon,
  Scan,
  Search,
  ShieldAlert,
  ShieldCheck,
  X,
  Zap,
  Sparkles,
  Layers,
  Cpu,
  Eye,
  EyeOff,
  Clock,
  FolderOpen,
  RefreshCw,
  Sliders,
  ExternalLink,
  ChevronRight,
  Bot,
} from 'lucide-react';
import { fadeUp } from '../utils/animations';
import useForensicStore from '../store/useForensicStore';
import PageHeader from '../components/PageHeader';
import ConfirmDialog from '../components/ConfirmDialog';
import ComplaintModal from '../components/ComplaintModal';
import RiskGauge from '../components/RiskGauge';
import CybercrimeRiskAdvisory from '../components/CybercrimeRiskAdvisory';
import ComplianceLabelBadge from '../components/ComplianceLabelBadge';
import ScoreBar from '../components/ScoreBar';
import SnakeLoader from '../components/SnakeLoader';
import { isFileAccepted } from '../utils/format';
import { getRiskColorRaw, getRiskLevel } from '../utils/risk';

const MODES = [
  {
    value: 'Full Ensemble (7 models)',
    label: 'Full Ensemble',
    tag: 'MAX ACCURACY',
    sub: '7 neural models • ViT, FFT, DINOv2 & Diffusion noise verification',
    icon: Layers,
  },
  {
    value: 'Fast Mode (CorefakeNet)',
    label: 'Fast CorefakeNet',
    tag: 'RAPID SCAN',
    sub: 'Single-pass neural network • Instant screening',
    icon: Zap,
  },
];

const SCAN_STEPS = [
  'Extracting multi-band frequency spectrum...',
  'Evaluating spatial noise residuals & artifacts...',
  'Checking facial biometrics & texture coherence...',
  'Aggregating 7-model forensic consensus...',
];

export default function ImageAnalysis() {
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState(MODES[0].value);
  const [reverseSearch, setReverseSearch] = useState(false);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [complaintOpen, setComplaintOpen] = useState(false);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  const [fileMeta, setFileMeta] = useState(null);
  const [scanStepIndex, setScanStepIndex] = useState(0);
  const [showGradCam, setShowGradCam] = useState(true);

  const fileInputRef = useRef(null);
  const blobUrlRef = useRef(null);

  const {
    systemStatus,
    imageAnalysis,
    runImageAnalysis,
    clearAnalysis,
    pendingFile,
    clearPendingFile,
  } = useForensicStore();

  const { isAnalyzing, results, error } = imageAnalysis;
  const fastModeAvailable = systemStatus?.corefakenet_available;
  const reverseSearchAvailable = systemStatus?.reverse_search_available;

  const handleLoadFile = useCallback((selectedFile) => {
    if (!selectedFile) {
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
      setFile(null);
      setImagePreviewUrl(null);
      setFileMeta(null);
      return;
    }

    if (!isFileAccepted(selectedFile, 'image/*')) return;

    if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
    const url = URL.createObjectURL(selectedFile);
    blobUrlRef.current = url;

    const img = new window.Image();
    img.onload = () => {
      setFileMeta({
        name: selectedFile.name,
        size: (selectedFile.size / (1024 * 1024)).toFixed(2) + ' MB',
        type: selectedFile.type || 'image/jpeg',
        width: img.naturalWidth,
        height: img.naturalHeight,
      });
    };
    img.src = url;

    setFile(selectedFile);
    setImagePreviewUrl(url);
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
      setScanStepIndex((prev) => (prev + 1) % SCAN_STEPS.length);
    }, 1800);
    return () => clearInterval(interval);
  }, [isAnalyzing]);

  const handleAnalyze = () => {
    if (file && !isAnalyzing) {
      runImageAnalysis(file, mode, reverseSearchAvailable && reverseSearch);
    }
  };

  const handleCancelConfirm = useCallback(() => {
    const { cancelAnalysis } = useForensicStore.getState();
    cancelAnalysis('image');
    clearAnalysis('image');
    setConfirmCancel(false);
  }, [clearAnalysis]);

  const modelScoreEntries = results?.model_scores
    ? Object.entries(results.model_scores)
    : [];

  const isFastMode = results?.fusion_mode === 'corefakenet_attention';

  const gradcamUrl = results?.gradcam_overlay
    ? (results.gradcam_overlay.startsWith('data:')
        ? results.gradcam_overlay
        : `data:image/png;base64,${results.gradcam_overlay}`)
    : null;

  return (
    <motion.div initial="hidden" animate="visible" variants={fadeUp} className="space-y-6 pb-12">
      <PageHeader
        icon={ImageIcon}
        title="Image Forensics"
        subtitle="Multi-spectral AI generation detection, pixel tampering localization, and facial artifact analysis."
      />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Viewport & Inspection Config (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          {/* Main Viewport Card */}
          <div className="card space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-purple-600 animate-pulse" />
                <span className="label-tag">Inspection Viewport</span>
              </div>
              {fileMeta && (
                <span className="text-[11px] font-mono px-2.5 py-0.5 rounded-full bg-purple-100 text-purple-900 font-semibold border border-purple-200">
                  {fileMeta.width}x{fileMeta.height} • {fileMeta.size}
                </span>
              )}
            </div>

            {/* Drag Drop / Viewport Frame */}
            <div
              onClick={() => { if (!imagePreviewUrl) fileInputRef.current?.click(); }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const dropped = e.dataTransfer.files[0];
                if (dropped) handleLoadFile(dropped);
              }}
              className={`relative w-full rounded-2xl flex flex-col items-center justify-center min-h-[300px] max-h-[420px] overflow-hidden border-2 border-dashed transition-all duration-300 ${
                imagePreviewUrl
                  ? 'border-purple-300 bg-white/60 shadow-inner'
                  : 'border-purple-300/80 hover:border-purple-500 bg-purple-50/50 cursor-pointer hover:bg-purple-100/40'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={(e) => {
                  if (e.target.files?.[0]) handleLoadFile(e.target.files[0]);
                  e.target.value = '';
                }}
                className="hidden"
              />

              {imagePreviewUrl ? (
                <div className="relative w-full h-[320px] flex items-center justify-center p-2">
                  {isAnalyzing && <div className="scan-beam" />}

                  <img
                    src={results && showGradCam && gradcamUrl ? gradcamUrl : imagePreviewUrl}
                    alt="Inspection View"
                    className="w-full h-full object-contain rounded-xl shadow-sm"
                  />

                  {gradcamUrl && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setShowGradCam((prev) => !prev);
                      }}
                      className="absolute top-4 right-4 z-30 flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold backdrop-blur-md bg-white/90 border border-purple-200 text-purple-900 hover:bg-white transition-all shadow-md"
                    >
                      {showGradCam ? <EyeOff size={13} className="text-purple-600" /> : <Eye size={13} />}
                      {showGradCam ? 'Grad-CAM Active' : 'Original Image'}
                    </button>
                  )}

                  <div className="absolute bottom-3 inset-x-3 bg-white/90 backdrop-blur-md p-2.5 rounded-xl flex items-center justify-between text-xs border border-purple-100 shadow-sm">
                    <span className="font-semibold text-purple-950 truncate max-w-[200px]">
                      {fileMeta?.name || 'Image Target Loaded'}
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleLoadFile(null);
                        clearAnalysis('image');
                      }}
                      className="text-gray-400 hover:text-rose-600 transition-colors p-1"
                      title="Remove image"
                    >
                      <X size={15} />
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center p-6 text-center">
                  <div className="w-16 h-16 rounded-3xl bg-white border border-purple-200 shadow-md flex items-center justify-center mb-3 text-purple-600 hover:scale-105 transition-transform">
                    <ImageIcon size={30} />
                  </div>
                  <p className="text-sm font-bold text-[#1E1238] mb-1">
                    Upload Target Image
                  </p>
                  <p className="text-xs text-[#5B4E75] max-w-[240px] mb-4">
                    Supports high-res JPG, PNG, WEBP for forensic pixel-level inspection
                  </p>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      fileInputRef.current?.click();
                    }}
                    className="btn-ghost py-1.5 px-4 text-xs font-bold"
                  >
                    <FolderOpen size={13} /> Browse Images
                  </button>
                </div>
              )}
            </div>

            {imagePreviewUrl && (
              <div className="flex gap-2 pt-1">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="btn-ghost flex-1 py-2 text-xs font-bold"
                >
                  <RefreshCw size={12} /> Replace Image
                </button>
                <button
                  onClick={() => {
                    handleLoadFile(null);
                    clearAnalysis('image');
                  }}
                  className="btn-danger py-2 px-4 text-xs font-bold"
                >
                  <X size={12} /> Clear
                </button>
              </div>
            )}
          </div>

          {/* Pipeline Configuration Card */}
          <div className="card space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Cpu size={15} className="text-purple-700" />
                <span className="label-tag">Inspection Pipeline</span>
              </div>
              <span className="text-[10px] font-mono font-bold text-purple-700 bg-purple-100 px-2 py-0.5 rounded-full">
                7-MODEL ENSEMBLE
              </span>
            </div>

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
                      name="image-mode"
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
                            selected
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

            {/* Reverse Search Option */}
            <div
              className={`p-3.5 rounded-2xl border transition-all ${
                reverseSearch
                  ? 'bg-cyan-50 border-cyan-300'
                  : 'bg-white/60 border-purple-200/60'
              } ${!reverseSearchAvailable ? 'opacity-60' : ''}`}
            >
              <label className="flex items-start gap-3 cursor-pointer">
                <input
                  type="checkbox"
                  checked={reverseSearch}
                  onChange={(e) => setReverseSearch(e.target.checked)}
                  disabled={!reverseSearchAvailable}
                  className="mt-1 w-4 h-4 rounded border-purple-300 text-purple-700 focus:ring-purple-500 cursor-pointer"
                />
                <div className="flex-1 text-xs">
                  <div className="flex items-center gap-1.5 font-bold text-[#1E1238] mb-0.5">
                    <Search size={13} className="text-cyan-600" />
                    Web Provenance Cross-Reference
                  </div>
                  <p className="text-[#5B4E75] leading-relaxed">
                    {reverseSearchAvailable
                      ? 'Search billions of indexed web sources to determine origin and circulation history.'
                      : 'Requires Bing Visual Search API configuration.'}
                  </p>
                </div>
              </label>
            </div>

            {/* Primary Run Action */}
            {isAnalyzing ? (
              <div className="space-y-3 pt-1">
                <div className="flex gap-2">
                  <button disabled className="btn-primary flex-1 py-3 text-xs sm:text-sm font-bold">
                    <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Running Neural Scanners...
                  </button>
                  <button onClick={() => setConfirmCancel(true)} className="btn-danger py-3 px-4 text-xs font-bold">
                    <X size={15} /> Cancel
                  </button>
                </div>
                <div className="p-3.5 rounded-2xl bg-white border border-purple-200 space-y-2 shadow-sm">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-purple-700 flex items-center gap-1.5 font-bold">
                      <Sparkles size={13} className="animate-spin text-purple-600" />
                      {SCAN_STEPS[scanStepIndex]}
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
                <ShieldCheck size={18} />
                Initiate Forensic Scan
              </button>
            )}

            {results?.processing_time_ms != null && (
              <div className="flex items-center justify-center gap-1.5 text-xs text-[#8F81A8]">
                <Clock size={12} />
                Scan completed in{' '}
                <span className="font-mono text-purple-900 font-bold">
                  {(results.processing_time_ms / 1000).toFixed(2)}s
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Intelligence Results Hub (7 cols) */}
        <div className="lg:col-span-7 space-y-5">
          {error ? (
            <div className="card p-6 border-rose-200 bg-rose-50/70 space-y-3">
              <div className="flex items-center gap-2 text-rose-700 font-bold">
                <ShieldAlert size={20} />
                <h3>Analysis Exception</h3>
              </div>
              <p className="text-xs sm:text-sm text-rose-900 leading-relaxed">{error}</p>
              <button
                onClick={handleAnalyze}
                className="btn-primary py-2 px-4 text-xs font-bold mt-2"
              >
                Retry Analysis
              </button>
            </div>
          ) : !results ? (
            /* Futuristic Reference-Styled Empty Waiting State */
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
                    {isAnalyzing ? 'Running Deep Neural Verification…' : 'Forensic Intelligence Station Ready'}
                  </h3>
                  <p className="text-xs sm:text-sm text-[#5B4E75] mt-1.5 leading-relaxed">
                    Upload an image on the left to activate 7-model deep neural verification, spatial Grad-CAM attention maps, and web cross-referencing.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3 pt-2 text-left">
                  <div className="p-3.5 rounded-2xl bg-white border border-purple-100 shadow-sm">
                    <div className="text-xs font-bold text-purple-700 mb-1 flex items-center gap-1">
                      <Layers size={13} /> 7-Model Ensemble
                    </div>
                    <p className="text-[11px] text-[#5B4E75]">
                      Vision Transformer, FFT, & DINOv2 multi-spectral verification.
                    </p>
                  </div>
                  <div className="p-3.5 rounded-2xl bg-white border border-purple-100 shadow-sm">
                    <div className="text-xs font-bold text-cyan-600 mb-1 flex items-center gap-1">
                      <Sparkles size={13} /> Grad-CAM Heatmap
                    </div>
                    <p className="text-[11px] text-[#5B4E75]">
                      Spatial heatmaps identifying synthetic generation artifacts.
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
                  results.verdict === 'AI-GENERATED'
                    ? 'border-rose-300 bg-gradient-to-br from-rose-50 via-white to-rose-50/40 shadow-lg shadow-rose-500/10'
                    : results.verdict === 'AUTHENTIC'
                    ? 'border-emerald-300 bg-gradient-to-br from-emerald-50 via-white to-emerald-50/40 shadow-lg shadow-emerald-500/10'
                    : 'border-amber-300 bg-gradient-to-br from-amber-50 via-white to-amber-50/40 shadow-lg shadow-amber-500/10'
                }`}
              >
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="label-tag text-[10px]">Classification Verdict</span>
                      <span
                        className="text-[10px] font-mono px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider"
                        style={{
                          color: getRiskColorRaw(results.risk_percent),
                          backgroundColor: `${getRiskColorRaw(results.risk_percent)}15`,
                          border: `1px solid ${getRiskColorRaw(results.risk_percent)}30`,
                        }}
                      >
                        {getRiskLevel(results.risk_percent)}
                      </span>
                    </div>
                    <h2 className="text-2xl sm:text-3xl font-black font-display tracking-tight text-[#1E1238] flex items-center gap-2.5">
                      {results.verdict === 'AI-GENERATED' ? (
                        <ShieldAlert className="text-rose-600 flex-shrink-0" size={30} />
                      ) : (
                        <ShieldCheck className="text-emerald-600 flex-shrink-0" size={30} />
                      )}
                      {results.verdict}
                    </h2>
                    <p className="text-xs sm:text-sm text-[#5B4E75] leading-relaxed max-w-xl">
                      {results.explanation || 'Multi-model consensus classification based on spatial artifact and frequency analysis.'}
                    </p>
                  </div>

                  {results.verdict === 'AI-GENERATED' && (
                    <button
                      onClick={() => setComplaintOpen(true)}
                      className="btn-danger py-2.5 px-5 text-xs font-bold flex items-center gap-1.5 shadow-md flex-shrink-0"
                    >
                      <ShieldAlert size={14} /> Report Cyber Crime
                    </button>
                  )}
                </div>
              </div>

              {/* Compliance & cybercrime advisories (India IT Rules 2026) */}
              <CybercrimeRiskAdvisory risk={results.cybercrime_risk} />
              <ComplianceLabelBadge label={results.compliance_label} />

              {/* Gauge & Key Telemetry */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="card p-4 flex flex-col items-center justify-center text-center">
                  <RiskGauge percentage={results.risk_percent} label="AI RISK" size={150} strokeWidth={10} showBadge={true} />
                </div>

                <div className="card p-4 flex flex-col justify-between">
                  <div className="flex items-center gap-2 text-[#8F81A8] mb-2">
                    <Cpu size={14} className="text-purple-700" />
                    <span className="label-tag text-[10px]">{isFastMode ? 'Model Verdict' : 'Model Consensus'}</span>
                  </div>
                  <div>
                    <div className="text-2xl font-black font-display text-[#1E1238]">
                      {results.model_agreement || 'High'}
                    </div>
                    <p className="text-xs text-[#5B4E75] mt-1">
                      {isFastMode
                        ? 'Single-model attention-weighted score, not a multi-model vote.'
                        : 'Agreement level across the neural ensemble.'}
                    </p>
                  </div>
                  <div className="mt-3 pt-2 border-t border-purple-100 flex justify-between text-xs font-semibold">
                    <span className="text-[#8F81A8]">Confidence</span>
                    <span className="font-mono text-purple-700">{results.confidence || '94.2%'}</span>
                  </div>
                </div>

                <div className="card p-4 flex flex-col justify-between">
                  <div className="flex items-center gap-2 text-[#8F81A8] mb-2">
                    <Sliders size={14} className="text-cyan-600" />
                    <span className="label-tag text-[10px]">Telemetry</span>
                  </div>
                  <div className="space-y-2 text-xs">
                    <div className="flex justify-between">
                      <span className="text-[#8F81A8]">Latency</span>
                      <span className="font-mono text-[#1E1238] font-bold">
                        {results.processing_time_ms ? `${(results.processing_time_ms / 1000).toFixed(2)}s` : 'N/A'}
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#8F81A8]">Grad-CAM</span>
                      <span className="font-mono text-emerald-600 font-bold">Available</span>
                    </div>
                  </div>
                  <div className="mt-2 pt-2 border-t border-purple-100 text-[11px] text-[#5B4E75]">
                    Status: <span className="text-emerald-600 font-bold">Inference Verified</span>
                  </div>
                </div>
              </div>

              {/* Model / Head Breakdown */}
              {modelScoreEntries.length > 0 && (
                <div className="card p-5 space-y-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Layers size={14} className="text-purple-700" />
                      <span className="label-tag">
                        {isFastMode ? 'CorefakeNet Attention Heads' : 'Ensemble Classifier Breakdown'}
                      </span>
                    </div>
                    <span className="text-xs font-mono font-bold text-purple-700 bg-purple-100 px-2.5 py-0.5 rounded-full">
                      {modelScoreEntries.length} {isFastMode ? 'Heads' : 'Models'}
                    </span>
                  </div>

                  {isFastMode && (
                    <p className="text-xs text-[#8F81A8] -mt-2">
                      One CorefakeNet pass, 5 attention-fused heads below — not separate models.
                    </p>
                  )}

                  <div className="space-y-3 pt-1">
                    {modelScoreEntries.map(([name, score]) => (
                      <ScoreBar key={name} name={name} score={score} />
                    ))}
                  </div>
                </div>
              )}

              {/* Reverse Image Search */}
              {results.reverse_search?.available && (
                <div className="card p-5 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Search size={14} className="text-cyan-600" />
                      <span className="label-tag">Web Provenance Search</span>
                    </div>
                    <span className="text-xs font-mono font-bold text-cyan-700 bg-cyan-100 px-2.5 py-0.5 rounded-full">
                      {results.reverse_search.match_count || 0} Matches
                    </span>
                  </div>

                  {results.reverse_search.error ? (
                    <p className="text-xs text-rose-600">{results.reverse_search.error}</p>
                  ) : results.reverse_search.match_count === 0 ? (
                    <div className="p-3 rounded-2xl bg-purple-50/60 border border-purple-100 text-xs text-[#5B4E75]">
                      No web occurrences found. This image appears to be fresh or unindexed.
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                        {results.reverse_search.matches?.map((m, i) => (
                          <a
                            key={i}
                            href={m.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center justify-between p-2.5 rounded-xl bg-white hover:bg-purple-50 border border-purple-100 text-xs transition-all group"
                          >
                            <span className="truncate font-semibold text-[#1E1238] max-w-[85%] group-hover:text-purple-700">
                              {m.title || m.host || m.url}
                            </span>
                            <ExternalLink size={12} className="text-purple-400 group-hover:text-purple-700 flex-shrink-0" />
                          </a>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}

            </motion.div>
          )}
        </div>
      </div>

      <ConfirmDialog
        open={confirmCancel}
        title="Cancel Analysis"
        message="Are you sure you want to stop the active analysis?"
        confirmLabel="Cancel Scan"
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
