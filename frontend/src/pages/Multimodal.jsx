import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Layers, Activity, Zap } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import RiskGauge from '../components/RiskGauge';
import ScoreBar from '../components/ScoreBar';
import VerdictCard from '../components/VerdictCard';
import useForensicStore from '../store/useForensicStore';

export default function Multimodal() {
  const [image, setImage] = useState(null);
  const [video, setVideo] = useState(null);
  const [audio, setAudio] = useState(null);

  const { multimodalAnalysis, runMultimodalAnalysis } = useForensicStore();
  const { isAnalyzing, results, error } = multimodalAnalysis;

  const handleAnalyze = () => {
    if (!image && !video && !audio) return;
    runMultimodalAnalysis(image, video, audio);
  };

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <header className="mb-8">
        <h2 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-accent-warning to-accent-danger">
          Multimodal Fusion
        </h2>
        <p className="text-text-muted mt-2">Upload multiple facets of media for a comprehensively compiled risk score.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-4">
          <UploadZone onFileSelect={setImage} accept="image/*" label="Image (Optional)" />
        </div>
        <div className="space-y-4">
          <UploadZone onFileSelect={setVideo} accept="video/*" label="Video (Optional)" />
        </div>
        <div className="space-y-4">
          <UploadZone onFileSelect={setAudio} accept="audio/*" label="Audio (Optional)" />
        </div>
      </div>

      <div className="flex justify-center my-6">
        <button
          onClick={handleAnalyze}
          disabled={(!image && !video && !audio) || isAnalyzing}
          className={`w-[400px] max-w-full py-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all duration-300
            ${((!image && !video && !audio) || isAnalyzing)
              ? 'bg-[rgba(255,255,255,0.05)] text-text-muted cursor-not-allowed'
              : 'btn-primary shadow-glow-accent hover:scale-105'}`}
        >
          {isAnalyzing ? (
            <><Activity className="animate-spin" size={20} /> <span className="animate-pulse">Fusing Modalities...</span></>
          ) : (
            <><Zap size={20} /> INITIATE MULTIMODAL FUSION</>
          )}
        </button>
      </div>

      {error && (
        <div className="p-4 bg-accent-danger/10 border border-accent-danger rounded-xl text-accent-danger text-center">
          {error}
        </div>
      )}

      {results && (
        <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} className="glass-card p-8">
          <h3 className="text-center font-bold text-lg mb-8 tracking-wider text-text-primary">FUSION INTELLIGENCE REPORT</h3>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-12 items-center">
            <div className="flex flex-col items-center">
              <RiskGauge
                percentage={results.data?.risk_score || results.data?.risk_percent || results.risk_percentage || 0}
                label="Aggregated Risk"
                size={220}
              />

              {results.risk_label && (
                <div className="w-full text-center py-2 mt-2 bg-[rgba(255,255,255,0.03)] rounded-lg text-xs font-semibold tracking-wide border border-border-subtle shadow-inner px-2">
                  {results.risk_label}
                </div>
              )}

              <div className="w-full mt-6">
                <VerdictCard verdict={results.verdict || results.data?.verdict} />
              </div>
            </div>

            <div className="space-y-4 p-6 bg-background rounded-xl border border-border-subtle">
              <h4 className="text-xs font-bold text-text-secondary uppercase tracking-widest mb-4">Modality Contributions</h4>

              {results.data?.modality_scores && Object.entries(results.data.modality_scores).map(([name, score]) => (
                <ScoreBar key={name} name={name.toUpperCase()} score={score} />
              ))}

              <div className="mt-6 pt-6 border-t border-[rgba(255,255,255,0.05)]">
                <p className="text-xs text-text-muted font-mono whitespace-pre-wrap leading-relaxed">
                  {results.data?.explanation || JSON.stringify(results.data, null, 2)}
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
