import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Mic, Activity, Volume2 } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import RiskGauge from '../components/RiskGauge';
import VerdictCard from '../components/VerdictCard';
import { forensicApi } from '../services/api';

export default function AudioAnalysis() {
  const [file, setFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  const handleAnalyze = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setError(null);
    try {
      const data = await forensicApi.analyzeAudio(file);
      if (data.success) {
        setResults(data);
      } else {
        setError(data.error || "Analysis failed");
      }
    } catch (err) {
      setError(err.response?.data?.error || err.message || "An error occurred");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <header className="mb-8">
        <h2 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-accent-cyan to-accent-green">
          Audio Analysis
        </h2>
        <p className="text-text-muted mt-2">Voice cloning and synthetic speech detection using frequency analysis.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Controls */}
        <div className="space-y-6">
          <UploadZone onFileSelect={setFile} accept="audio/*" label="Upload Audio (WAV/MP3)" />
          
          <button 
            onClick={handleAnalyze} 
            disabled={!file || isAnalyzing}
            className={`w-full py-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all duration-300
              ${(!file || isAnalyzing) 
                ? 'bg-[rgba(255,255,255,0.05)] text-text-muted cursor-not-allowed' 
                : 'bg-gradient-to-r from-accent-green to-accent-cyan text-[#0A0E1A] shadow-[0_0_20px_rgba(16,185,129,0.3)] hover:scale-[0.98]'}`}
          >
            {isAnalyzing ? (
              <><Activity className="animate-spin" size={20} /> <span className="animate-pulse">Processing Audio...</span></>
            ) : (
              <><Volume2 size={20} /> EXTRACT SPECTROGRAMS</>
            )}
          </button>
        </div>

        {/* Results */}
        <div className="glass-card p-6 min-h-[400px]">
          <h3 className="text-sm font-bold text-text-secondary uppercase tracking-wider mb-6 flex items-center gap-2">
            <Mic size={16} /> Spectral Intelligence
          </h3>
          
          {error ? (
            <div className="p-4 bg-[rgba(236,72,153,0.1)] border border-accent-pink rounded-lg text-accent-pink text-sm">
              {error}
            </div>
          ) : results ? (
            <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center">
              <div className="flex gap-10">
                <RiskGauge percentage={results.risk_percentage || results.data?.risk_percent || 0} label="AI Risk" size={160} />
                <RiskGauge percentage={results.authenticity_percentage || results.data?.authenticity_score || 0} label="Authenticity" size={160} />
              </div>
              
              {results.risk_label && (
                <div className="w-full text-center py-2 mt-4 bg-[rgba(255,255,255,0.03)] rounded-lg text-sm font-semibold tracking-wide border border-border-subtle shadow-inner">
                  {results.risk_label}
                </div>
              )}
              
              <div className="w-full mt-6">
                <VerdictCard verdict={results.verdict || results.data?.verdict} />
              </div>

              {(results.details || results.data?.explanation) && (
                <div className="w-full mt-6 p-4 bg-[#0A0E1A] rounded-lg border border-border-subtle">
                  <pre className="text-xs text-text-muted font-mono whitespace-pre-wrap leading-relaxed">
                    {results.details || results.data.explanation}
                  </pre>
                </div>
              )}
            </motion.div>
          ) : (
            <div className="h-full flex flex-col items-center justify-center text-text-muted opacity-30 mt-20">
              <Mic size={48} className="mb-4" />
              <p>Waiting for audio input</p>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
