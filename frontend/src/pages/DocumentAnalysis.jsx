import React, { useState, useCallback, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  FileSearch,
  IdCard,
  Search,
  ShieldAlert,
  ShieldCheck,
  X,
  Check,
  AlertTriangle,
  Sparkles,
  ScanLine,
  Fingerprint,
  Clock,
  ExternalLink,
  FolderOpen,
  RefreshCw,
} from 'lucide-react';
import { fadeUp, staggerFadeUp } from '../utils/animations';
import useForensicStore from '../store/useForensicStore';
import PageHeader from '../components/PageHeader';
import ConfirmDialog from '../components/ConfirmDialog';
import ComplaintModal from '../components/ComplaintModal';
import RiskGauge from '../components/RiskGauge';
import SnakeLoader from '../components/SnakeLoader';
import { isFileAccepted } from '../utils/format';
import { getRiskColorRaw, getRiskLevel } from '../utils/risk';

const CHECK_LABELS = {
  ai_generation: 'AI Generation',
  tampering: 'Tampering (ELA)',
  copy_move: 'Copy-Move',
  metadata: 'Metadata',
  id_number: 'ID Number',
  c2pa: 'C2PA Provenance',
};

const CHECK_OK_VALUES = new Set(['Not detected', 'Analyzed', 'Valid format', 'Verified']);

const ID_TYPES = [
  { value: '', label: 'None / Other document' },
  { value: 'aadhaar', label: 'Aadhaar' },
  { value: 'pan', label: 'PAN' },
  { value: 'voter_id', label: 'Voter ID (EPIC)' },
];

const ID_PLACEHOLDERS = {
  aadhaar: 'e.g. 234567890124 (12 digits)',
  pan: 'e.g. ABCDE1234F',
  voter_id: 'e.g. ABC1234567',
};

const SCAN_STEPS = [
  'Running Error-Level Analysis across compression blocks...',
  'Checking noise-grid consistency & copy-move regions...',
  'Reading EXIF metadata & C2PA provenance chain...',
  'Cross-checking ID number format & checksum...',
];

const VERDICT_STYLE = {
  'AI-GENERATED': {
    border: 'border-rose-300',
    bg: 'bg-gradient-to-br from-rose-50 via-white to-rose-50/40',
    shadow: 'shadow-lg shadow-rose-500/10',
    Icon: ShieldAlert,
    iconColor: 'text-rose-600',
  },
  MANIPULATED: {
    border: 'border-amber-300',
    bg: 'bg-gradient-to-br from-amber-50 via-white to-amber-50/40',
    shadow: 'shadow-lg shadow-amber-500/10',
    Icon: AlertTriangle,
    iconColor: 'text-amber-600',
  },
  AUTHENTIC: {
    border: 'border-emerald-300',
    bg: 'bg-gradient-to-br from-emerald-50 via-white to-emerald-50/40',
    shadow: 'shadow-lg shadow-emerald-500/10',
    Icon: ShieldCheck,
    iconColor: 'text-emerald-600',
  },
};

function CheckBadge({ name, value, index }) {
  const ok = CHECK_OK_VALUES.has(value);
  return (
    <motion.div
      custom={index}
      variants={staggerFadeUp}
      initial="hidden"
      animate="visible"
      className={`flex items-center justify-between gap-2 px-3.5 py-2.5 rounded-2xl border text-sm transition-all hover:-translate-y-0.5 ${
        ok
          ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
          : 'bg-amber-50 border-amber-200 text-amber-800'
      }`}
    >
      <span className="flex items-center gap-2 font-semibold text-[#1E1238]">
        {ok ? <Check size={13} className="text-emerald-600" /> : <AlertTriangle size={13} className="text-amber-600" />}
        {CHECK_LABELS[name] || name}
      </span>
      <span className="text-xs font-bold font-mono">{value}</span>
    </motion.div>
  );
}

export default function DocumentAnalysis() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [fileMeta, setFileMeta] = useState(null);
  const [idType, setIdType] = useState('');
  const [idNumber, setIdNumber] = useState('');
  const [reverseSearch, setReverseSearch] = useState(false);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [complaintOpen, setComplaintOpen] = useState(false);
  const [scanStepIndex, setScanStepIndex] = useState(0);

  const { systemStatus, documentAnalysis, runDocumentAnalysis, clearAnalysis } = useForensicStore();
  const { isAnalyzing, results, error } = documentAnalysis;
  const reverseSearchAvailable = systemStatus?.reverse_search_available;

  const handleLoadFile = useCallback((selectedFile) => {
    if (!selectedFile) {
      setFile(null);
      setPreviewUrl(null);
      setFileMeta(null);
      return;
    }
    if (!isFileAccepted(selectedFile, 'image/*')) return;

    const url = URL.createObjectURL(selectedFile);
    setFile(selectedFile);
    setPreviewUrl(url);
    setFileMeta({
      name: selectedFile.name,
      size: (selectedFile.size / (1024 * 1024)).toFixed(2) + ' MB',
      type: selectedFile.type || 'image/jpeg',
    });
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
    if (file) runDocumentAnalysis(file, idType, idNumber, reverseSearchAvailable && reverseSearch);
  };

  const handleCancelRequest = useCallback(() => setConfirmCancel(true), []);
  const handleCancelConfirm = useCallback(() => {
    const { cancelAnalysis } = useForensicStore.getState();
    cancelAnalysis('document');
    clearAnalysis('document');
    setConfirmCancel(false);
  }, [clearAnalysis]);

  const checkEntries = results?.checks ? Object.entries(results.checks) : [];
  const verdictStyle = VERDICT_STYLE[results?.verdict] || VERDICT_STYLE.AUTHENTIC;
  const VerdictIcon = verdictStyle.Icon;

  return (
    <motion.div initial="hidden" animate="visible" variants={fadeUp} className="space-y-6 pb-12">
      <PageHeader
        icon={FileSearch}
        title="Document Forensics"
        subtitle="Screen IDs, receipts, certificates, and scanned documents for AI generation or tampering."
      />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Viewport & Config (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          <div className="card space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-purple-600 animate-pulse" />
                <span className="label-tag">Document Viewport</span>
              </div>
              {fileMeta && (
                <span className="text-[11px] font-mono px-2.5 py-0.5 rounded-full bg-purple-100 text-purple-900 font-semibold border border-purple-200">
                  {fileMeta.size} • {fileMeta.type.replace('image/', '').toUpperCase()}
                </span>
              )}
            </div>

            <div
              onClick={() => { if (!previewUrl) document.getElementById('doc-file-input')?.click(); }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const dropped = e.dataTransfer.files[0];
                if (dropped) handleLoadFile(dropped);
              }}
              className={`relative w-full rounded-2xl flex flex-col items-center justify-center min-h-[280px] max-h-[400px] overflow-hidden border-2 border-dashed transition-all duration-300 ${
                previewUrl
                  ? 'border-purple-300 bg-white/60 shadow-inner'
                  : 'border-purple-300/80 hover:border-purple-500 bg-purple-50/50 cursor-pointer hover:bg-purple-100/40'
              }`}
            >
              <input
                id="doc-file-input"
                type="file"
                accept="image/*"
                onChange={(e) => {
                  if (e.target.files?.[0]) handleLoadFile(e.target.files[0]);
                  e.target.value = '';
                }}
                className="hidden"
              />

              {previewUrl ? (
                <div className="relative w-full h-[300px] flex items-center justify-center p-2">
                  {isAnalyzing && <div className="scan-beam" />}
                  <img src={previewUrl} alt="Document preview" className="w-full h-full object-contain rounded-xl shadow-sm" />
                  <div className="absolute bottom-3 inset-x-3 bg-white/90 backdrop-blur-md p-2.5 rounded-xl flex items-center justify-between text-xs border border-purple-100 shadow-sm">
                    <span className="font-semibold text-purple-950 truncate max-w-[200px]">
                      {fileMeta?.name || 'Document loaded'}
                    </span>
                    <button
                      onClick={(e) => { e.stopPropagation(); handleLoadFile(null); clearAnalysis('document'); }}
                      className="text-gray-400 hover:text-rose-600 transition-colors p-1"
                      title="Remove document"
                    >
                      <X size={15} />
                    </button>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center p-6 text-center">
                  <div className="w-16 h-16 rounded-3xl bg-white border border-purple-200 shadow-md flex items-center justify-center mb-3 text-purple-600 hover:scale-105 transition-transform">
                    <FileSearch size={30} />
                  </div>
                  <p className="text-sm font-bold text-[#1E1238] mb-1">Upload Document or ID</p>
                  <p className="text-xs text-[#5B4E75] max-w-[240px] mb-4">
                    Supports JPG, PNG, WEBP for forensic tamper & AI-generation screening
                  </p>
                  <button
                    type="button"
                    onClick={(e) => { e.stopPropagation(); document.getElementById('doc-file-input')?.click(); }}
                    className="btn-ghost py-1.5 px-4 text-xs font-bold"
                  >
                    <FolderOpen size={13} /> Browse Files
                  </button>
                </div>
              )}
            </div>

            {previewUrl && (
              <div className="flex gap-2 pt-1">
                <button
                  onClick={() => document.getElementById('doc-file-input')?.click()}
                  className="btn-ghost flex-1 py-2 text-xs font-bold"
                >
                  <RefreshCw size={12} /> Replace File
                </button>
                <button
                  onClick={() => { handleLoadFile(null); clearAnalysis('document'); }}
                  className="btn-danger py-2 px-4 text-xs font-bold"
                >
                  <X size={12} /> Clear
                </button>
              </div>
            )}

            <div className="p-3 rounded-2xl bg-purple-50/70 border border-purple-100 text-xs text-[#5B4E75] leading-relaxed">
              Heuristic forensic checks (error-level analysis, noise consistency,
              copy-move detection, metadata, C2PA). AI-generation detection relies
              on EXIF/C2PA evidence only — a neural detector was tested but produced
              false positives on real government IDs, so it isn't used here.
              Not a trained document classifier yet — treat results as a first pass.
            </div>
          </div>

          {/* Config Card */}
          <div className="card space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <IdCard size={15} className="text-purple-700" />
                <span className="label-tag">ID Number Check (Optional)</span>
              </div>
            </div>

            <div className="space-y-3">
              <div>
                <label className="text-xs mb-1 block font-semibold text-[#5B4E75]">ID Type</label>
                <select
                  value={idType}
                  onChange={(e) => { setIdType(e.target.value); setIdNumber(''); }}
                  className="field-input text-sm"
                >
                  {ID_TYPES.map((t) => (
                    <option key={t.value} value={t.value}>{t.label}</option>
                  ))}
                </select>
              </div>

              {idType && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}>
                  <label className="text-xs mb-1 block font-semibold text-[#5B4E75]">
                    {ID_TYPES.find((t) => t.value === idType)?.label} Number
                  </label>
                  <input
                    type="text"
                    value={idNumber}
                    onChange={(e) => setIdNumber(e.target.value)}
                    placeholder={ID_PLACEHOLDERS[idType]}
                    className="field-input text-sm font-mono"
                  />
                  <p className="text-xs mt-1.5 text-[#8F81A8]">
                    {idType === 'aadhaar'
                      ? 'Checked against the real Verhoeff checksum UIDAI uses — not a guess.'
                      : 'Checked against the standard format only — no public checksum exists for this ID type.'}
                  </p>
                </motion.div>
              )}
            </div>

            <div
              className={`p-3.5 rounded-2xl border transition-all ${
                reverseSearch ? 'bg-cyan-50 border-cyan-300' : 'bg-white/60 border-purple-200/60'
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
                    Reverse Image Search
                  </div>
                  <p className="text-[#5B4E75] leading-relaxed">
                    {reverseSearchAvailable
                      ? 'Cross-reference against the public web (Bing Visual Search). Sends the image to a third-party provider.'
                      : 'Requires Bing Visual Search API configuration.'}
                  </p>
                </div>
              </label>
            </div>

            {isAnalyzing ? (
              <div className="space-y-3 pt-1">
                <div className="flex gap-2">
                  <button disabled className="btn-primary flex-1 py-3 text-xs sm:text-sm font-bold">
                    <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Analyzing Document...
                  </button>
                  <button onClick={handleCancelRequest} className="btn-danger py-3 px-4 text-xs font-bold">
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
                Run Document Analysis
              </button>
            )}

            {results?.processing_time_ms != null && (
              <div className="flex items-center justify-center gap-1.5 text-xs text-[#8F81A8]">
                <Clock size={12} />
                Completed in{' '}
                <span className="font-mono text-purple-900 font-bold">
                  {(results.processing_time_ms / 1000).toFixed(2)}s
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Results Hub (7 cols) */}
        <div className="lg:col-span-7 space-y-5">
          {error ? (
            <div className="card p-6 border-rose-200 bg-rose-50/70 space-y-3">
              <div className="flex items-center gap-2 text-rose-700 font-bold">
                <ShieldAlert size={20} />
                <h3>Document Analysis Error</h3>
              </div>
              <p className="text-xs sm:text-sm text-rose-900 leading-relaxed">{error}</p>
              <button onClick={handleAnalyze} className="btn-primary py-2 px-4 text-xs font-bold mt-2">
                Retry Scan
              </button>
            </div>
          ) : !results ? (
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
                    {isAnalyzing ? 'Scanning Document Forensics…' : 'Document Forensics Station Ready'}
                  </h3>
                  <p className="text-xs sm:text-sm text-[#5B4E75] mt-1.5 leading-relaxed">
                    Upload an ID, receipt, or certificate to check for AI generation, digital
                    tampering, and provenance credentials.
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-3 pt-2 text-left">
                  <div className="p-3.5 rounded-2xl bg-white border border-purple-100 shadow-sm">
                    <div className="text-xs font-bold text-purple-700 mb-1 flex items-center gap-1">
                      <ScanLine size={13} /> ELA & Copy-Move
                    </div>
                    <p className="text-[11px] text-[#5B4E75]">
                      Classical forensic detection of splices and recompression seams.
                    </p>
                  </div>
                  <div className="p-3.5 rounded-2xl bg-white border border-purple-100 shadow-sm">
                    <div className="text-xs font-bold text-cyan-600 mb-1 flex items-center gap-1">
                      <Fingerprint size={13} /> ID Checksum
                    </div>
                    <p className="text-[11px] text-[#5B4E75]">
                      Real Verhoeff checksum validation for Aadhaar numbers.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-5">
              {/* Verdict Banner */}
              <div className={`card p-6 border relative overflow-hidden ${verdictStyle.border} ${verdictStyle.bg} ${verdictStyle.shadow}`}>
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                  <div className="space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="label-tag text-[10px]">Document Classification</span>
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
                      <VerdictIcon className={`${verdictStyle.iconColor} flex-shrink-0`} size={30} />
                      {results.verdict}
                    </h2>
                    <p className="text-xs sm:text-sm text-[#5B4E75] leading-relaxed max-w-xl">
                      {results.primary_finding || 'Combined classical-forensic and neural classification.'}
                    </p>
                  </div>

                  {results.verdict && results.verdict !== 'AUTHENTIC' && (
                    <button
                      onClick={() => setComplaintOpen(true)}
                      className="btn-danger py-2.5 px-5 text-xs font-bold flex items-center gap-1.5 shadow-md flex-shrink-0"
                    >
                      <ShieldAlert size={14} /> Report Cyber Crime
                    </button>
                  )}
                </div>
              </div>

              {/* Gauge & Confidence */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <div className="card p-4 flex flex-col items-center justify-center text-center">
                  <RiskGauge percentage={results.risk_percent} label="RISK SCORE" size={150} strokeWidth={10} showBadge={true} />
                </div>

                <div className="card p-4 flex flex-col justify-between">
                  <div className="flex items-center gap-2 text-[#8F81A8] mb-2">
                    <Fingerprint size={14} className="text-purple-700" />
                    <span className="label-tag text-[10px]">Confidence</span>
                  </div>
                  <div>
                    <div className="text-2xl font-black font-display text-[#1E1238]">
                      {results.confidence || 'MEDIUM'}
                    </div>
                    <p className="text-xs text-[#5B4E75] mt-1">
                      Combined classical-forensic and neural signal strength.
                    </p>
                  </div>
                </div>

                <div className="card p-4 flex flex-col justify-between">
                  <div className="flex items-center gap-2 text-[#8F81A8] mb-2">
                    <ScanLine size={14} className="text-cyan-600" />
                    <span className="label-tag text-[10px]">Signals</span>
                  </div>
                  <div className="space-y-1.5 text-xs">
                    <div className="flex justify-between">
                      <span className="text-[#8F81A8]">AI-generated</span>
                      <span className="font-mono text-[#1E1238] font-bold">{((results.ai_generated_score || 0) * 100).toFixed(0)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-[#8F81A8]">Manipulated</span>
                      <span className="font-mono text-[#1E1238] font-bold">{((results.manipulation_score || 0) * 100).toFixed(0)}%</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Visual Checks Grid */}
              {checkEntries.length > 0 && (
                <div className="card p-5 space-y-3">
                  <div className="flex items-center gap-2">
                    <ScanLine size={14} className="text-purple-700" />
                    <span className="label-tag">Visual Analysis</span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                    {checkEntries.map(([name, value], i) => (
                      <CheckBadge key={name} name={name} value={value} index={i} />
                    ))}
                  </div>
                </div>
              )}

              {/* ID Validation */}
              {results.id_validation && (
                <div
                  className={`card p-4 border text-sm ${
                    results.id_validation.valid ? 'bg-emerald-50/70 border-emerald-200' : 'bg-amber-50/70 border-amber-200'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    {results.id_validation.valid
                      ? <Check size={14} className="text-emerald-600" />
                      : <AlertTriangle size={14} className="text-amber-600" />}
                    <span className="font-bold text-[#1E1238]">
                      {results.id_validation.id_label} Number: {results.id_validation.valid ? 'Valid' : 'Invalid'}
                    </span>
                  </div>
                  <p className="text-xs text-[#5B4E75] leading-relaxed pl-6">{results.id_validation.reason}</p>
                </div>
              )}

              {/* C2PA */}
              {results.c2pa && (
                <div
                  className={`card p-4 border text-sm ${
                    results.c2pa.ai_generated_signal
                      ? 'bg-rose-50/70 border-rose-200'
                      : results.c2pa.validation_state === 'Invalid'
                      ? 'bg-amber-50/70 border-amber-200'
                      : results.c2pa.valid
                      ? 'bg-emerald-50/70 border-emerald-200'
                      : 'bg-purple-50/60 border-purple-100'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    {results.c2pa.ai_generated_signal || results.c2pa.validation_state === 'Invalid' ? (
                      <AlertTriangle size={14} className={results.c2pa.ai_generated_signal ? 'text-rose-600' : 'text-amber-600'} />
                    ) : (
                      <Check size={14} className="text-emerald-600" />
                    )}
                    <span className="font-bold text-[#1E1238]">
                      C2PA Content Credentials:{' '}
                      {results.c2pa.ai_generated_signal
                        ? 'AI generation declared'
                        : results.c2pa.validation_state === 'Invalid'
                        ? 'Tampered'
                        : results.c2pa.valid
                        ? 'Verified'
                        : 'Present, unverified'}
                    </span>
                  </div>
                  <p className="text-xs text-[#5B4E75] leading-relaxed pl-6">
                    {results.c2pa.ai_generated_signal
                      ? 'The embedded manifest itself declares this content was produced by a generative AI tool (digitalSourceType: trainedAlgorithmicMedia).'
                      : results.c2pa.validation_state === 'Invalid'
                      ? 'A C2PA manifest is present but its signature failed validation — the provenance chain was altered after signing.'
                      : results.c2pa.valid
                      ? 'A signed, valid provenance chain was found with no AI-generation declaration.'
                      : 'A C2PA manifest is present but could not be fully validated.'}
                    {results.c2pa.generator && <> Generator: <strong>{results.c2pa.generator}</strong>.</>}
                  </p>
                </div>
              )}

              {/* ELA Evidence */}
              {results.evidence?.ela_map && (
                <div className="card p-5 space-y-3">
                  <div className="flex items-center gap-2">
                    <ScanLine size={14} className="text-purple-700" />
                    <span className="label-tag">Error Level Analysis (Evidence)</span>
                  </div>
                  <img
                    src={results.evidence.ela_map}
                    alt="Error level analysis heatmap"
                    className="w-full rounded-2xl border border-purple-100 shadow-sm"
                  />
                </div>
              )}

              {/* Reverse Search */}
              {results.reverse_search?.available && (
                <div className="card p-5 space-y-3">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <Search size={14} className="text-cyan-600" />
                      <span className="label-tag">Reverse Image Search</span>
                    </div>
                    <span className="text-xs font-mono font-bold text-cyan-700 bg-cyan-100 px-2.5 py-0.5 rounded-full">
                      {results.reverse_search.match_count || 0} Matches
                    </span>
                  </div>
                  {results.reverse_search.error ? (
                    <p className="text-xs text-rose-600">Lookup failed: {results.reverse_search.error}</p>
                  ) : results.reverse_search.match_count === 0 ? (
                    <div className="p-3 rounded-2xl bg-purple-50/60 border border-purple-100 text-xs text-[#5B4E75]">
                      No matching pages found on the public web.
                    </div>
                  ) : (
                    <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
                      {results.reverse_search.matches.map((m, i) => (
                        <a
                          key={i}
                          href={m.url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="flex items-center justify-between p-2.5 rounded-xl bg-white hover:bg-purple-50 border border-purple-100 text-xs transition-all group"
                        >
                          <span className="truncate font-semibold text-[#1E1238] max-w-[85%] group-hover:text-purple-700">
                            {m.title || m.host}
                          </span>
                          <ExternalLink size={12} className="text-purple-400 group-hover:text-purple-700 flex-shrink-0" />
                        </a>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* EXIF Findings */}
              {results.exif?.findings?.length > 0 && (
                <div className="card p-5 space-y-2">
                  <div className="flex items-center gap-2">
                    <Fingerprint size={14} className="text-purple-700" />
                    <span className="label-tag">Metadata Findings</span>
                  </div>
                  <ul className="text-sm text-[#5B4E75] space-y-1.5 list-disc list-inside pl-1">
                    {results.exif.findings.map((f, i) => (
                      <li key={i}>{f}</li>
                    ))}
                  </ul>
                </div>
              )}
            </motion.div>
          )}
        </div>
      </div>

      <ConfirmDialog
        open={confirmCancel}
        title="Cancel Analysis"
        message="The current analysis is still running. Are you sure you want to cancel?"
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
