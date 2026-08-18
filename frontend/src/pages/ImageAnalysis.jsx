import React, { useState, useEffect, useCallback } from 'react';
import { Image, Scan, ShieldCheck, X } from 'lucide-react';
import useForensicStore from '../store/useForensicStore';
import PageHeader from '../components/PageHeader';
import ConfirmDialog from '../components/ConfirmDialog';
import UploadZone from '../components/UploadZone';
import RiskGauge from '../components/RiskGauge';
import ScoreBar from '../components/ScoreBar';
import VerdictCard from '../components/VerdictCard';
import HeatmapViewer from '../components/HeatmapViewer';
import IndeterminateProgress from '../components/IndeterminateProgress';

const MODES = [
  {
    value: 'Full Ensemble (7 models)',
    label: 'Full Ensemble',
    sub: '7 models -- High accuracy',
  },
  {
    value: 'Fast Mode (CorefakeNet)',
    label: 'Fast Mode',
    sub: 'Single-pass -- CorefakeNet',
  },
];

export default function ImageAnalysis() {
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState(MODES[0].value);
  const [confirmCancel, setConfirmCancel] = useState(false);

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

  useEffect(() => {
    if (pendingFile) {
      setFile(pendingFile);
      clearPendingFile();
    }
  }, [pendingFile, clearPendingFile]);

  const handleAnalyze = () => {
    if (file) runImageAnalysis(file, mode);
  };

  const handleCancelRequest = useCallback(() => setConfirmCancel(true), []);
  const handleCancelConfirm = useCallback(() => {
    const { cancelAnalysis } = useForensicStore.getState();
    cancelAnalysis('image');
    clearAnalysis('image');
    setConfirmCancel(false);
  }, [clearAnalysis]);

  const modelScoreEntries = results?.model_scores
    ? Object.entries(results.model_scores)
    : [];

  return (
    <div className="space-y-6">
      <PageHeader
        icon={Image}
        title="Image Analysis"
        subtitle="Detect manipulation, AI generation, and face-swapping in still images."
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Left column: Upload + Mode + Analyze */}
        <div className="lg:col-span-1 space-y-4">
          <UploadZone
            onFileSelect={setFile}
            accept="image/*"
            label="Drop image or click to browse"
            initialFile={pendingFile}
          />

          {/* Mode selector */}
          <div className="card">
            <div className="flex items-center gap-2 mb-3">
              <Scan size={13} className="text-text-3" />
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
                      name="analysis-mode"
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

          {/* Analyze / Cancel buttons */}
          {isAnalyzing ? (
            <div className="space-y-3">
              <div className="flex gap-2">
                <button disabled className="btn-primary flex-1 py-3 text-sm">
                  <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Analyzing...
                </button>
                <button onClick={handleCancelRequest} className="btn-danger py-3 px-4 text-sm">
                  <X size={15} />
                  Cancel
                </button>
              </div>
              <IndeterminateProgress label="Running forensic models…" />
            </div>
          ) : (
            <button
              onClick={handleAnalyze}
              disabled={!file}
              className="btn-primary w-full py-3 text-sm"
            >
              <ShieldCheck size={15} />
              Run Forensic Analysis
            </button>
          )}

          {results?.processing_time_ms != null && (
            <p className="text-xs text-center text-text-3">
              Completed in {(results.processing_time_ms / 1000).toFixed(1)}s
            </p>
          )}
        </div>

        {/* Right column: Results panel */}
        <div className="lg:col-span-2">
          <div className="card">
            <div className="flex items-center gap-2 mb-3">
              <ShieldCheck size={13} className="text-text-3" />
              <span className="label-tag">Intelligence Report</span>
            </div>

            {error ? (
              <div
                role="alert"
                className="p-3 rounded-lg text-sm bg-risk-criticalDim text-risk-critical border border-[rgba(251,113,133,0.20)]"
              >
                {error}
              </div>
            ) : !results ? (
              <div className="flex items-center justify-center gap-2 py-8 text-text-3">
                <ShieldCheck size={16} className="opacity-30" />
                <p className="text-sm">Upload an image and run analysis to see results.</p>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="flex flex-col items-center">
                  <RiskGauge percentage={results.risk_percent} />

                  <div className="flex flex-wrap justify-center gap-4 mt-2">
                    {results.confidence && (
                      <span className="text-xs text-text-2">
                        Confidence: <strong className="text-text-1">{results.confidence}</strong>
                      </span>
                    )}
                    {results.model_agreement && (
                      <span className="text-xs text-text-2">
                        Agreement: <strong className="text-text-1">{results.model_agreement}</strong>
                      </span>
                    )}
                  </div>
                </div>

                {modelScoreEntries.length > 0 && (
                  <div>
                    <span className="label-tag mb-2 block">Model Scores</span>
                    {modelScoreEntries.map(([name, score]) => (
                      <ScoreBar key={name} name={name} score={score} />
                    ))}
                  </div>
                )}

                <VerdictCard verdict={results.verdict} riskScore={results.risk_percent} />

                {results.explanation && (
                  <div className="p-3 rounded-lg bg-bg-inset border border-border-dim">
                    <span className="label-tag mb-1.5 block">Explanation</span>
                    <p className="text-sm font-mono whitespace-pre-wrap leading-relaxed text-text-2">
                      {results.explanation}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      <HeatmapViewer
        originalFile={file}
        gradcamBase64={results?.gradcam_image}
      />

      <ConfirmDialog
        open={confirmCancel}
        title="Cancel Analysis"
        message="The current analysis is still running. Are you sure you want to cancel?"
        confirmLabel="Cancel Analysis"
        onConfirm={handleCancelConfirm}
        onCancel={() => setConfirmCancel(false)}
      />
    </div>
  );
}
