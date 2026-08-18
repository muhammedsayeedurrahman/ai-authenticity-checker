import React, { useState, useEffect, useCallback } from 'react';
import { Mic, Volume2, ShieldCheck, X } from 'lucide-react';
import PageHeader from '../components/PageHeader';
import ConfirmDialog from '../components/ConfirmDialog';
import UploadZone from '../components/UploadZone';
import RiskGauge from '../components/RiskGauge';
import VerdictCard from '../components/VerdictCard';
import AudioWaveform from '../components/AudioWaveform';
import IndeterminateProgress from '../components/IndeterminateProgress';
import useForensicStore from '../store/useForensicStore';

export default function AudioAnalysis() {
  const [file, setFile] = useState(null);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const { audioAnalysis, runAudioAnalysis, clearAnalysis, pendingFile, clearPendingFile } = useForensicStore();
  const { isAnalyzing, results, error } = audioAnalysis;

  useEffect(() => {
    if (pendingFile) {
      setFile(pendingFile);
      clearPendingFile();
    }
  }, [pendingFile, clearPendingFile]);

  const handleAnalyze = () => {
    if (file) runAudioAnalysis(file);
  };

  const handleCancelConfirm = useCallback(() => {
    const { cancelAnalysis } = useForensicStore.getState();
    cancelAnalysis('audio');
    clearAnalysis('audio');
    setConfirmCancel(false);
  }, [clearAnalysis]);

  return (
    <div className="space-y-6">
      <PageHeader
        icon={Mic}
        title="Audio Analysis"
        subtitle="Voice cloning and synthetic speech detection using spectral analysis."
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Left: Upload */}
        <div className="lg:col-span-1 space-y-4">
          <UploadZone onFileSelect={setFile} accept="audio/*" label="Upload Audio (WAV/MP3)" />

          <AudioWaveform file={file} />

          <div className="flex gap-2">
            <button
              onClick={handleAnalyze}
              disabled={!file || isAnalyzing}
              className="btn-primary flex-1 py-3"
            >
              {isAnalyzing ? (
                <>
                  <span className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Processing Audio...
                </>
              ) : (
                <><Volume2 size={15} /> Analyze Audio</>
              )}
            </button>
            {isAnalyzing && (
              <button onClick={() => setConfirmCancel(true)} className="btn-danger px-3" title="Cancel">
                <X size={15} />
              </button>
            )}
          </div>
          {isAnalyzing && (
            <IndeterminateProgress label="Analyzing spectral patterns…" />
          )}
        </div>

        {/* Right: Results */}
        <div className="lg:col-span-2 card min-h-[360px]">
          <div className="flex items-center gap-2 mb-4">
            <Mic size={14} className="text-text-3" />
            <span className="label-tag">Spectral Analysis</span>
          </div>

          {error ? (
            <div
              role="alert"
              className="p-3 rounded-lg text-sm bg-risk-criticalDim text-risk-critical border border-[rgba(251,113,133,0.20)]"
            >
              {error}
            </div>
          ) : results ? (
            <div className="flex flex-col items-center space-y-4">
              <div className="flex flex-wrap justify-center gap-6">
                <RiskGauge percentage={results.risk_percent || 0} label="AI Risk" size={150} />
                <RiskGauge percentage={results.authenticity_percentage || 0} label="Authenticity" size={150} />
              </div>

              <div className="w-full">
                <VerdictCard verdict={results.verdict} riskScore={results.risk_percent} />
              </div>

              {results.manipulation_type && (
                <div className="w-full inset-panel p-3">
                  <p className="label-tag mb-1">Manipulation Type</p>
                  <p className="text-sm font-medium text-text-1">
                    {results.manipulation_type}
                  </p>
                </div>
              )}

              {results.explanation && (
                <div className="w-full inset-panel p-3">
                  <p className="label-tag mb-1">Details</p>
                  <p className="text-sm leading-relaxed text-text-2">
                    {results.explanation}
                  </p>
                </div>
              )}

              {results.duration_sec > 0 && (
                <div className="w-full grid grid-cols-2 gap-3">
                  <div className="inset-panel p-3 text-center">
                    <p className="text-base font-bold font-mono text-text-1">
                      {results.duration_sec.toFixed(1)}s
                    </p>
                    <p className="text-xs text-text-3">Duration</p>
                  </div>
                  <div className="inset-panel p-3 text-center">
                    <p className="text-base font-bold font-mono text-text-1">
                      {results.segments_analyzed || 0}
                    </p>
                    <p className="text-xs text-text-3">Segments</p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-16 space-y-2 text-text-3">
              <ShieldCheck size={22} className="opacity-30" />
              <p className="text-sm">Waiting for audio input</p>
            </div>
          )}
        </div>
      </div>

      <ConfirmDialog
        open={confirmCancel}
        title="Cancel Analysis"
        message="Audio processing is still running. Are you sure you want to cancel?"
        confirmLabel="Cancel Analysis"
        onConfirm={handleCancelConfirm}
        onCancel={() => setConfirmCancel(false)}
      />
    </div>
  );
}
