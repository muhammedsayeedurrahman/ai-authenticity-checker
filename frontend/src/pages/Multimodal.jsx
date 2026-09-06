import React, { useState, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Layers, Zap, ShieldCheck, ShieldAlert, X, Sparkles, Waves, Clock } from 'lucide-react';
import { fadeUp, staggerFadeUp } from '../utils/animations';
import PageHeader from '../components/PageHeader';
import ConfirmDialog from '../components/ConfirmDialog';
import ComplaintModal from '../components/ComplaintModal';
import UploadZone from '../components/UploadZone';
import RiskGauge from '../components/RiskGauge';
import CybercrimeRiskAdvisory from '../components/CybercrimeRiskAdvisory';
import ComplianceLabelBadge from '../components/ComplianceLabelBadge';
import ScoreBar from '../components/ScoreBar';
import FusionVisualizer from '../components/FusionVisualizer';
import SnakeLoader from '../components/SnakeLoader';
import useForensicStore from '../store/useForensicStore';
import { getRiskColorRaw, getRiskLevel } from '../utils/risk';

const FUSION_STEPS = [
  'Running per-modality forensic ensembles in parallel...',
  'Cross-referencing image, video, and audio signals...',
  'Computing weighted cross-modal fusion score...',
  'Identifying which modalities triggered the verdict...',
];

const VERDICT_STYLE = {
  'AI-GENERATED': {
    border: 'border-rose-300',
    bg: 'bg-gradient-to-br from-rose-50 via-white to-rose-50/40',
    shadow: 'shadow-lg shadow-rose-500/10',
    Icon: ShieldAlert,
    iconColor: 'text-rose-600',
  },
  AUTHENTIC: {
    border: 'border-emerald-300',
    bg: 'bg-gradient-to-br from-emerald-50 via-white to-emerald-50/40',
    shadow: 'shadow-lg shadow-emerald-500/10',
    Icon: ShieldCheck,
    iconColor: 'text-emerald-600',
  },
};

export default function Multimodal() {
  const [image, setImage] = useState(null);
  const [video, setVideo] = useState(null);
  const [audio, setAudio] = useState(null);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [complaintOpen, setComplaintOpen] = useState(false);

  const { multimodalAnalysis, runMultimodalAnalysis, clearAnalysis } = useForensicStore();
  const { isAnalyzing, results, error } = multimodalAnalysis;

  const hasInput = image || video || audio;
  const verdictStyle = VERDICT_STYLE[results?.verdict] || VERDICT_STYLE.AUTHENTIC;
  const VerdictIcon = verdictStyle.Icon;

  const [scanStepIndex, setScanStepIndex] = useState(0);
  React.useEffect(() => {
    if (!isAnalyzing) { setScanStepIndex(0); return; }
    const interval = setInterval(() => {
      setScanStepIndex((prev) => (prev + 1) % FUSION_STEPS.length);
    }, 2000);
    return () => clearInterval(interval);
  }, [isAnalyzing]);

  const handleAnalyze = () => {
    if (hasInput) runMultimodalAnalysis(image, video, audio);
  };

  const handleCancelConfirm = useCallback(() => {
    const { cancelAnalysis } = useForensicStore.getState();
    cancelAnalysis('multimodal');
    clearAnalysis('multimodal');
    setConfirmCancel(false);
  }, [clearAnalysis]);

  return (
    <motion.div initial="hidden" animate="visible" variants={fadeUp} className="space-y-6 pb-12">
      <PageHeader
        icon={Layers}
        title="Multimodal Fusion"
        subtitle="Upload image, video, and audio together for a cross-modal risk assessment."
      />

      {/* Upload Grid */}
      <div className="card space-y-4">
        <div className="flex items-center gap-2">
          <Waves size={15} className="text-purple-700" />
          <span className="label-tag">Modality Inputs</span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
          <UploadZone onFileSelect={setImage} accept="image/*" label="Image (Optional)" />
          <UploadZone onFileSelect={setVideo} accept="video/*" label="Video (Optional)" />
          <UploadZone onFileSelect={setAudio} accept="audio/*" label="Audio (Optional)" />
        </div>

        <FusionVisualizer image={image} video={video} audio={audio} isAnalyzing={isAnalyzing} />

        {isAnalyzing ? (
          <div className="space-y-3">
            <div className="flex gap-2">
              <button disabled className="btn-primary flex-1 py-3.5 text-sm font-bold">
                <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Fusing Modalities...
              </button>
              <button onClick={() => setConfirmCancel(true)} className="btn-danger py-3.5 px-4 text-xs font-bold">
                <X size={15} /> Cancel
              </button>
            </div>
            <div className="p-3.5 rounded-2xl bg-white border border-purple-200 space-y-2.5 shadow-sm">
              <div className="flex items-center justify-center py-1">
                <SnakeLoader
                  width={7}
                  speed={70}
                  snakeColor="#6D28D9"
                  appleColor="#EC4899"
                  className="gap-px"
                  dotClassName="size-[5px] rounded-[1px]"
                />
              </div>
              <div className="flex items-center justify-between text-xs">
                <span className="text-purple-700 flex items-center gap-1.5 font-bold">
                  <Sparkles size={13} className="animate-spin text-purple-600" />
                  {FUSION_STEPS[scanStepIndex]}
                </span>
                <span className="text-[10px] font-mono text-purple-500 font-bold">STEP {scanStepIndex + 1}/4</span>
              </div>
              <div className="progress-indeterminate-track" />
            </div>
          </div>
        ) : (
          <button
            onClick={handleAnalyze}
            disabled={!hasInput}
            className="btn-primary w-full py-3.5 text-sm font-bold shadow-md shadow-purple-900/20"
          >
            <Zap size={18} />
            Run Multimodal Analysis
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

      {/* Error */}
      {error && (
        <div className="card p-6 border-rose-200 bg-rose-50/70 space-y-3">
          <div className="flex items-center gap-2 text-rose-700 font-bold">
            <ShieldAlert size={20} />
            <h3>Fusion Analysis Error</h3>
          </div>
          <p className="text-xs sm:text-sm text-rose-900 leading-relaxed">{error}</p>
          <button onClick={handleAnalyze} className="btn-primary py-2 px-4 text-xs font-bold mt-2">
            Retry Analysis
          </button>
        </div>
      )}

      {/* Results */}
      {results && (
        <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-5">
          {/* Verdict Banner */}
          <div className={`card p-6 border relative overflow-hidden ${verdictStyle.border} ${verdictStyle.bg} ${verdictStyle.shadow}`}>
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="label-tag text-[10px]">Fusion Classification</span>
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
                  {results.explanation || 'Weighted cross-modal fusion across submitted media.'}
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

            {results.clean_modalities?.length > 0 && results.flagged_modalities?.length > 0 && (
              <div className="mt-4 p-3 rounded-2xl bg-white/70 border border-purple-100 text-xs leading-relaxed">
                <p>
                  <span className="text-rose-600 font-bold">
                    Flagged: {results.flagged_modalities.join(', ')}
                  </span>
                  {'  —  '}
                  <span className="text-emerald-600 font-bold">
                    Authentic: {results.clean_modalities.join(', ')}
                  </span>
                </p>
                <p className="mt-1 text-[#8F81A8]">
                  The verdict above reflects the combined submission — not every uploaded
                  file is necessarily AI-generated.
                </p>
              </div>
            )}
          </div>

          {/* Compliance & cybercrime advisories (India IT Rules 2026) */}
          {results.cybercrime_risks?.map((risk, i) => (
            <CybercrimeRiskAdvisory key={risk.category + i} risk={risk} />
          ))}
          <ComplianceLabelBadge label={results.compliance_label} />

          {/* Gauge + Modality Contributions */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
            <div className="card p-5 flex flex-col items-center justify-center">
              <RiskGauge percentage={results.risk_percent || 0} label="Aggregated Risk" size={190} strokeWidth={11} showBadge={true} />
            </div>

            <div className="card p-5 space-y-4">
              <div className="flex items-center gap-2">
                <Layers size={14} className="text-purple-700" />
                <span className="label-tag">Modality Contributions</span>
              </div>

              <div>
                {results.modality_scores && Object.entries(results.modality_scores).map(([name, score], i) => (
                  score != null && (
                    <motion.div key={name} custom={i} variants={staggerFadeUp} initial="hidden" animate="visible">
                      <ScoreBar name={name.toUpperCase()} score={score} />
                    </motion.div>
                  )
                ))}
              </div>

              {results.fusion_weights && (
                <div className="pt-3 border-t border-purple-100">
                  <p className="label-tag mb-2 text-[10px]">Fusion Weights</p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(results.fusion_weights).map(([mod, w]) => (
                      <span
                        key={mod}
                        className="text-[11px] font-mono font-bold px-2.5 py-1 rounded-full bg-purple-100 text-purple-800 border border-purple-200"
                      >
                        {mod}: {(w * 100).toFixed(0)}%
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        </motion.div>
      )}

      {/* Empty state */}
      {!results && !error && !isAnalyzing && (
        <div className="card p-8 flex flex-col items-center justify-center text-center relative overflow-hidden">
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
              <h3 className="text-lg font-black text-[#1E1238]">Fusion Station Ready</h3>
              <p className="text-xs sm:text-sm text-[#5B4E75] mt-1.5 leading-relaxed">
                Upload any combination of image, video, and audio above — each modality
                is scored independently, then combined into one weighted verdict.
              </p>
            </div>
          </div>
        </div>
      )}

      <ConfirmDialog
        open={confirmCancel}
        title="Cancel Analysis"
        message="Multimodal fusion is still running. Are you sure you want to cancel?"
        confirmLabel="Cancel Analysis"
        onConfirm={handleCancelConfirm}
        onCancel={() => setConfirmCancel(false)}
      />

      <ComplaintModal
        open={complaintOpen}
        onClose={() => setComplaintOpen(false)}
        analysis={results}
        fileName={image?.name || video?.name || audio?.name}
      />
    </motion.div>
  );
}
