import React, { useState, useCallback } from 'react';
import { Layers, Zap, ShieldCheck, X } from 'lucide-react';
import PageHeader from '../components/PageHeader';
import ConfirmDialog from '../components/ConfirmDialog';
import UploadZone from '../components/UploadZone';
import RiskGauge from '../components/RiskGauge';
import ScoreBar from '../components/ScoreBar';
import VerdictCard from '../components/VerdictCard';
import FusionVisualizer from '../components/FusionVisualizer';
import IndeterminateProgress from '../components/IndeterminateProgress';
import useForensicStore from '../store/useForensicStore';

export default function Multimodal() {
  const [image, setImage] = useState(null);
  const [video, setVideo] = useState(null);
  const [audio, setAudio] = useState(null);
  const [confirmCancel, setConfirmCancel] = useState(false);

  const { multimodalAnalysis, runMultimodalAnalysis, clearAnalysis } = useForensicStore();
  const { isAnalyzing, results, error } = multimodalAnalysis;

  const hasInput = image || video || audio;

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
    <div className="space-y-6">
      <PageHeader
        icon={Layers}
        title="Multimodal Fusion"
        subtitle="Upload multiple media types for a cross-modal risk assessment."
      />

      {/* 3-column upload grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        <UploadZone onFileSelect={setImage} accept="image/*" label="Image (Optional)" />
        <UploadZone onFileSelect={setVideo} accept="video/*" label="Video (Optional)" />
        <UploadZone onFileSelect={setAudio} accept="audio/*" label="Audio (Optional)" />
      </div>

      <FusionVisualizer image={image} video={video} audio={audio} isAnalyzing={isAnalyzing} />

      {/* Analyze button */}
      <div className="flex justify-center gap-2">
        <button
          onClick={handleAnalyze}
          disabled={!hasInput || isAnalyzing}
          className="btn-primary max-w-sm w-full py-3"
        >
          {isAnalyzing ? (
            <>
              <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Fusing Modalities...
            </>
          ) : (
            <><Zap size={15} /> Run Multimodal Analysis</>
          )}
        </button>
        {isAnalyzing && (
          <button onClick={() => setConfirmCancel(true)} className="btn-danger px-3" title="Cancel">
            <X size={15} />
          </button>
        )}
      </div>

      {isAnalyzing && (
        <div className="max-w-sm mx-auto">
          <IndeterminateProgress label="Fusing modality signals…" />
        </div>
      )}

      {/* Error */}
      {error && (
        <div
          role="alert"
          className="p-3 rounded-lg text-sm text-center bg-risk-criticalDim text-risk-critical border border-[rgba(251,113,133,0.20)]"
        >
          {error}
        </div>
      )}

      {/* Results */}
      {results && (
        <div className="card">
          <p className="label-tag mb-5 text-center">Fusion Report</p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 items-start">
            {/* Left: Gauge + Verdict */}
            <div className="flex flex-col items-center space-y-4">
              <RiskGauge
                percentage={results.risk_percent || 0}
                label="Aggregated Risk"
                size={200}
              />
              <div className="w-full">
                <VerdictCard verdict={results.verdict} riskScore={results.risk_percent} />
              </div>
            </div>

            {/* Right: Modality contributions */}
            <div className="inset-panel p-5 space-y-4">
              <h4 className="label-tag">Modality Contributions</h4>

              {results.modality_scores && Object.entries(results.modality_scores).map(([name, score]) => (
                score != null && <ScoreBar key={name} name={name.toUpperCase()} score={score} />
              ))}

              {results.fusion_weights && (
                <div className="pt-3 border-t border-border-dim">
                  <p className="label-tag mb-2">Fusion Weights</p>
                  <div className="flex gap-3">
                    {Object.entries(results.fusion_weights).map(([mod, w]) => (
                      <span key={mod} className="text-xs font-mono text-text-2">
                        {mod}: {(w * 100).toFixed(0)}%
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {results.explanation && (
                <div className="pt-3 border-t border-border-dim">
                  <p className="text-sm leading-relaxed text-text-2">
                    {results.explanation}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Empty state */}
      {!results && !error && !isAnalyzing && (
        <div className="flex flex-col items-center py-12 text-text-3">
          <ShieldCheck size={24} className="opacity-30 mb-2" />
          <p className="text-sm">Upload at least one media file to begin analysis</p>
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
    </div>
  );
}
