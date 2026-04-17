import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Film, Activity, Settings2, Play } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import RiskGauge from '../components/RiskGauge';
import VerdictCard from '../components/VerdictCard';
import FrameTable from '../components/FrameTable';
import { forensicApi } from '../services/api';

export default function VideoAnalysis() {
  const [file, setFile] = useState(null);
  const [fps, setFps] = useState(6);
  const [aggregation, setAggregation] = useState('weighted_avg');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  
  const handleAnalyze = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setError(null);
    try {
      const data = await forensicApi.analyzeVideo(file, fps, aggregation);
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

  const videoBase64 = results?.gradcam_video;
  const videoUrl = videoBase64 ? `data:video/mp4;base64,${videoBase64}` : null;

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <header className="mb-8">
        <h2 className="text-3xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-text-secondary">
          Video Forensics
        </h2>
        <p className="text-text-muted mt-2">Frame-by-frame deepfake analysis with temporal consistency checking.</p>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-4 gap-6">
        {/* Left Column: Controls */}
        <div className="xl:col-span-1 space-y-6">
          <UploadZone onFileSelect={setFile} accept="video/*" label="Upload Video (MP4/AVI)" />
          
          <div className="glass-card p-4 space-y-5">
            <h3 className="text-sm font-bold text-text-secondary uppercase tracking-wider flex items-center gap-2">
              <Settings2 size={16} /> Parameters
            </h3>
            
            <div>
              <div className="flex justify-between text-xs text-text-primary mb-2">
                <span>Sampling FPS</span>
                <span className="font-mono text-accent-cyan">{fps} fps</span>
              </div>
              <input 
                type="range" min="1" max="15" step="0.5" 
                value={fps} onChange={(e) => setFps(Number(e.target.value))}
                className="w-full h-1 bg-[rgba(255,255,255,0.1)] rounded-lg appearance-none cursor-pointer accent-accent-cyan"
              />
            </div>
            
            <div>
              <label className="text-xs text-text-primary mb-2 block">Temporal Aggregation</label>
              <select 
                value={aggregation} onChange={(e) => setAggregation(e.target.value)}
                className="w-full bg-[rgba(0,0,0,0.3)] border border-border-subtle rounded-lg p-2 text-sm text-text-primary focus:outline-none focus:border-accent-cyan"
              >
                <option value="weighted_avg">Attention Weighted Avg</option>
                <option value="max">Max Peak Risk</option>
                <option value="average">Simple Average</option>
                <option value="majority">Majority Vote</option>
              </select>
            </div>
          </div>
          
          <button 
            onClick={handleAnalyze} 
            disabled={!file || isAnalyzing}
            className={`w-full py-4 rounded-xl font-bold flex items-center justify-center gap-2 transition-all duration-300
              ${(!file || isAnalyzing) 
                ? 'bg-[rgba(255,255,255,0.05)] text-text-muted cursor-not-allowed' 
                : 'btn-primary shadow-glow-violet'}`}
          >
            {isAnalyzing ? (
              <><Activity className="animate-spin" size={20} /> <span className="animate-pulse">Processing Frames...</span></>
            ) : (
              <><Play size={20} /> RUN ANALYSIS</>
            )}
          </button>
        </div>

        {/* Center/Right Column: Results & Playback */}
        <div className="xl:col-span-3 space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2 glass-card flex items-center justify-center min-h-[300px] overflow-hidden p-2">
              {videoUrl ? (
                <video src={videoUrl} controls autoPlay loop className="max-w-full max-h-full rounded-lg" />
              ) : (
                <div className="text-center text-text-muted">
                  <Film size={48} className="mx-auto mb-3 opacity-20" />
                  <p>GradCAM Heatmap Video</p>
                </div>
              )}
            </div>
            <div className="lg:col-span-1 glass-card p-4 flex flex-col justify-center">
              {error ? (
                <div className="p-4 bg-[rgba(236,72,153,0.1)] border border-accent-pink rounded-lg text-accent-pink text-sm">
                  {error}
                </div>
              ) : results ? (
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center">
                  <RiskGauge percentage={results.risk_percentage || results.data?.risk_percent || 0} label="Video Avg Risk" size={180} />
                  
                  {results.risk_label && (
                    <div className="w-full text-center py-2 mt-2 bg-[rgba(255,255,255,0.03)] rounded-lg text-[11px] font-semibold tracking-wide border border-border-subtle shadow-inner px-2">
                       {results.risk_label}
                    </div>
                  )}

                  <div className="w-full mt-2">
                    <VerdictCard verdict={results.verdict || results.data?.verdict} />
                  </div>
                </motion.div>
              ) : (
                <div className="text-center text-text-muted">
                  <p>Awaiting Results</p>
                </div>
              )}
            </div>
          </div>

          {(results?.frame_details) && (
            <div className="glass-card p-4">
               <h3 className="text-sm font-bold text-text-secondary uppercase tracking-wider mb-4">Frame Timeline Analysis</h3>
               <FrameTable framesRawStr={results.frame_details} />
            </div>
          )}
          
          {(results?.details || results?.data?.explanation) && (
            <div className="glass-card p-4">
              <h3 className="text-sm font-bold text-text-secondary uppercase tracking-wider mb-4">Intelligence Details</h3>
              <pre className="text-xs text-text-muted font-mono whitespace-pre-wrap leading-relaxed bg-[#0A0E1A] p-4 rounded-lg border border-border-subtle">
                {results.details || JSON.stringify(results.data?.temporal_analysis, null, 2) || results.data?.explanation}
              </pre>
            </div>
          )}
        </div>
      </div>
    </motion.div>
  );
}
