import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Activity, ShieldAlert, Cpu } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import RiskGauge from '../components/RiskGauge';
import ScoreBar from '../components/ScoreBar';
import VerdictCard from '../components/VerdictCard';
import HeatmapViewer from '../components/HeatmapViewer';
import { forensicApi } from '../services/api';
import useForensicStore from '../store/useForensicStore';

export default function ImageAnalysis() {
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState('Full Ensemble (7 models)');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  const { systemStatus } = useForensicStore();
  const fastModeAvailable = systemStatus?.corefakenet_ready || systemStatus?.corefakenet_available;

  const handleAnalyze = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setError(null);
    try {
      const data = await forensicApi.analyzeImage(file, mode);
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
        <h2 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-text-secondary">
          Image Analysis
        </h2>
        <p className="text-text-muted mt-2">Identify manipulation, AI generation, and face-swapping in still images.</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Left Column: Controls */}
        <div className="lg:col-span-1 space-y-6">
          <div className="glass-card p-4">
            <h3 className="text-sm font-bold text-text-secondary uppercase tracking-wider mb-4 flex items-center gap-2">
              <Cpu size={16} /> Analysis Mode
            </h3>
            <div className="space-y-3">
              <label className="flex items-center gap-3 p-3 rounded-lg border border-[rgba(0,240,255,0.2)] bg-[rgba(0,240,255,0.03)] cursor-pointer hover:bg-[rgba(0,240,255,0.06)] transition-colors">
                <input 
                  type="radio" 
                  name="mode" 
                  value="Full Ensemble (7 models)" 
                  checked={mode === 'Full Ensemble (7 models)'}
                  onChange={(e) => setMode(e.target.value)}
                  className="accent-accent-cyan"
                />
                <div className="flex flex-col">
                  <span className="text-sm font-bold text-white">Full Ensemble</span>
                  <span className="text-[10px] text-text-muted">High accuracy (7 models)</span>
                </div>
              </label>
              <label className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer hover:bg-background-card-hover transition-colors
                ${!fastModeAvailable ? 'opacity-50 grayscale cursor-not-allowed border-border-subtle' : 'border-border-subtle bg-background-card'}`}>
                <input 
                  type="radio" 
                  name="mode" 
                  value="Fast Mode (CorefakeNet)" 
                  checked={mode === 'Fast Mode (CorefakeNet)'}
                  onChange={(e) => fastModeAvailable && setMode(e.target.value)}
                  disabled={!fastModeAvailable}
                  className="accent-accent-violet"
                />
                <div className="flex flex-col">
                  <span className="text-sm font-bold text-white">Fast Mode</span>
                  <span className="text-[10px] text-text-muted">CorefakeNet Single-Pass</span>
                </div>
              </label>
            </div>
          </div>
          
          <UploadZone onFileSelect={setFile} label="Upload Image (JPG/PNG)" />
          
          <button 
            onClick={handleAnalyze} 
            disabled={!file || isAnalyzing}
            className={`w-full py-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all duration-300
              ${(!file || isAnalyzing) 
                ? 'bg-[rgba(255,255,255,0.05)] text-text-muted cursor-not-allowed' 
                : 'btn-primary shadow-glow-cyan'}`}
          >
            {isAnalyzing ? (
              <><Activity className="animate-spin" size={20} /> <span className="animate-pulse">Analyzing...</span></>
            ) : (
              <><ShieldAlert size={20} /> INITIALIZE ANALYSIS</>
            )}
          </button>
        </div>

        {/* Center Column: Visualization */}
        <div className="lg:col-span-2">
          <HeatmapViewer 
            originalFile={file} 
            gradcamBase64={results?.gradcam || results?.data?.gradcam_image} 
          />
        </div>

        {/* Right Column: Results */}
        <div className="lg:col-span-1 space-y-4">
          <div className="glass-card p-4 min-h-[400px]">
            <h3 className="text-sm font-bold text-text-secondary uppercase tracking-wider mb-2 flex items-center gap-2">
              <Activity size={16} /> Intelligence Report
            </h3>
            
            {error ? (
              <div className="p-4 bg-[rgba(236,72,153,0.1)] border border-accent-pink rounded-lg text-accent-pink text-sm mt-4">
                {error}
              </div>
            ) : !results ? (
              <div className="h-full flex flex-col items-center justify-center text-text-muted mt-20">
                <Activity size={32} className="opacity-20 mb-2" />
                <p className="text-sm">Awaiting Payload</p>
              </div>
            ) : (
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                <RiskGauge percentage={results.risk_percentage || results.data?.risk_percent || 0} />
                
                {results.risk_label && (
                  <div className="w-full text-center py-2 mb-2 bg-[rgba(255,255,255,0.03)] rounded-lg text-sm font-semibold tracking-wide border border-border-subtle shadow-inner">
                    {results.risk_label}
                  </div>
                )}
                
                <div className="mt-4 space-y-1">
                  {results.model_scores && Object.entries(results.model_scores).map(([name, score]) => (
                    <ScoreBar key={name} name={name} score={score} />
                  ))}
                  {results.data?.model_scores && Object.entries(results.data.model_scores).map(([name, score]) => (
                    <ScoreBar key={name} name={name} score={score} />
                  ))}
                </div>

                <VerdictCard verdict={results.verdict || results.data?.verdict} />
                
                {(results.details || results.data?.explanation) && (
                  <div className="mt-4 p-3 bg-[#0A0E1A] rounded-lg border border-border-subtle">
                    <p className="text-[10px] text-text-muted font-mono whitespace-pre-wrap leading-relaxed">
                      {results.details || results.data.explanation}
                    </p>
                  </div>
                )}
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
