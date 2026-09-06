import React, { useState, useEffect, useCallback, useRef } from 'react';
import { motion } from 'framer-motion';
import {
  Mic,
  Volume2,
  VolumeX,
  Play,
  Pause,
  ShieldCheck,
  ShieldAlert,
  X,
  Sparkles,
  Activity,
  Waves,
  Clock,
  AudioLines,
  FolderOpen,
  RefreshCw,
  Sliders,
  FileAudio,
} from 'lucide-react';
import { fadeUp } from '../utils/animations';
import PageHeader from '../components/PageHeader';
import ConfirmDialog from '../components/ConfirmDialog';
import ComplaintModal from '../components/ComplaintModal';
import RiskGauge from '../components/RiskGauge';
import CybercrimeRiskAdvisory from '../components/CybercrimeRiskAdvisory';
import ComplianceLabelBadge from '../components/ComplianceLabelBadge';
import SnakeLoader from '../components/SnakeLoader';
import useForensicStore from '../store/useForensicStore';
import { isFileAccepted } from '../utils/format';
import { getRiskColorRaw, getRiskLevel } from '../utils/risk';

const AUDIO_SCAN_STEPS = [
  'Extracting Mel-Frequency Cepstral Coefficients (MFCC)...',
  'Analyzing harmonic spectral anomalies & vocoder artifacts...',
  'Checking synthetic pitch modulation & breath continuity...',
  'Evaluating voice-biometric authenticity index...',
];

export default function AudioAnalysis() {
  const [file, setFile] = useState(null);
  const [confirmCancel, setConfirmCancel] = useState(false);
  const [complaintOpen, setComplaintOpen] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(1);
  const [isMuted, setIsMuted] = useState(false);
  const [audioMeta, setAudioMeta] = useState(null);
  const [waveformPeaks, setWaveformPeaks] = useState([]);
  const [, setIsDecodingAudio] = useState(false);
  const [scanStepIndex, setScanStepIndex] = useState(0);

  const fileInputRef = useRef(null);
  const audioElementRef = useRef(null);
  const blobUrlRef = useRef(null);

  const {
    audioAnalysis,
    runAudioAnalysis,
    clearAnalysis,
    pendingFile,
    clearPendingFile,
  } = useForensicStore();

  const { isAnalyzing, results, error } = audioAnalysis;

  const handleLoadFile = useCallback(async (selectedFile) => {
    if (!selectedFile) {
      if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
      blobUrlRef.current = null;
      setFile(null);
      setAudioMeta(null);
      setWaveformPeaks([]);
      setIsPlaying(false);
      setCurrentTime(0);
      setDuration(0);
      return;
    }

    if (!isFileAccepted(selectedFile, 'audio/*')) return;

    if (blobUrlRef.current) URL.revokeObjectURL(blobUrlRef.current);
    const url = URL.createObjectURL(selectedFile);
    blobUrlRef.current = url;

    setFile(selectedFile);
    setAudioMeta({
      name: selectedFile.name,
      size: (selectedFile.size / (1024 * 1024)).toFixed(2) + ' MB',
      type: selectedFile.type || 'audio/wav',
    });

    setIsDecodingAudio(true);
    try {
      const arrayBuffer = await selectedFile.arrayBuffer();
      const AudioContextCls = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioContextCls();
      const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

      setDuration(audioBuffer.duration);

      const raw = audioBuffer.getChannelData(0);
      const totalBars = 56;
      const blockSize = Math.max(1, Math.floor(raw.length / totalBars));
      const peaks = [];
      for (let i = 0; i < totalBars; i++) {
        const start = i * blockSize;
        let max = 0;
        for (let j = 0; j < blockSize && start + j < raw.length; j++) {
          const abs = Math.abs(raw[start + j]);
          if (abs > max) max = abs;
        }
        peaks.push(max);
      }
      const peakMax = Math.max(...peaks, 0.0001);
      const normalized = peaks.map((p) => Math.max(0.1, p / peakMax));
      setWaveformPeaks(normalized);
      await audioCtx.close();
    } catch (e) {
      console.warn('Waveform fallback:', e);
      setWaveformPeaks(Array.from({ length: 56 }, () => 0.2 + Math.random() * 0.6));
    } finally {
      setIsDecodingAudio(false);
    }
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
      setScanStepIndex((prev) => (prev + 1) % AUDIO_SCAN_STEPS.length);
    }, 1800);
    return () => clearInterval(interval);
  }, [isAnalyzing]);

  const togglePlay = () => {
    if (!audioElementRef.current) return;
    if (isPlaying) {
      audioElementRef.current.pause();
      setIsPlaying(false);
    } else {
      audioElementRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  };

  const handleTimeUpdate = () => {
    if (audioElementRef.current) {
      setCurrentTime(audioElementRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioElementRef.current) {
      setDuration(audioElementRef.current.duration || 0);
    }
  };

  const handleSeek = (e) => {
    const time = Number(e.target.value);
    setCurrentTime(time);
    if (audioElementRef.current) {
      audioElementRef.current.currentTime = time;
    }
  };

  const handleEnded = () => {
    setIsPlaying(false);
    setCurrentTime(0);
  };

  const formatTime = (secs) => {
    if (isNaN(secs) || secs < 0) return '00:00';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  const handleAnalyze = () => {
    if (file && !isAnalyzing) {
      runAudioAnalysis(file);
    }
  };

  const handleCancelConfirm = useCallback(() => {
    const { cancelAnalysis } = useForensicStore.getState();
    cancelAnalysis('audio');
    clearAnalysis('audio');
    setConfirmCancel(false);
  }, [clearAnalysis]);

  return (
    <motion.div initial="hidden" animate="visible" variants={fadeUp} className="space-y-6 pb-12">
      <PageHeader
        icon={Mic}
        title="Voice & Audio Forensics"
        subtitle="Synthetic voice clone detection, deepfake speech analysis, and acoustic biometric verification."
      />

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Workstation Player & Control (5 cols) */}
        <div className="lg:col-span-5 space-y-5">
          {/* Audio Workstation Player */}
          <div className="card space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan-500 animate-pulse" />
                <span className="label-tag">Acoustic Workstation</span>
              </div>
              {audioMeta && (
                <span className="text-[11px] font-mono px-2.5 py-0.5 rounded-full bg-purple-100 text-purple-900 font-semibold border border-purple-200">
                  {audioMeta.size} • {audioMeta.type.replace('audio/', '').toUpperCase()}
                </span>
              )}
            </div>

            {blobUrlRef.current && (
              <audio
                ref={audioElementRef}
                src={blobUrlRef.current}
                onTimeUpdate={handleTimeUpdate}
                onLoadedMetadata={handleLoadedMetadata}
                onEnded={handleEnded}
                className="hidden"
              />
            )}

            <div
              onClick={() => { if (!file) fileInputRef.current?.click(); }}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const dropped = e.dataTransfer.files[0];
                if (dropped) handleLoadFile(dropped);
              }}
              className={`relative w-full rounded-2xl flex flex-col items-center justify-center min-h-[220px] p-5 border-2 border-dashed transition-all duration-300 ${
                file
                  ? 'border-purple-300 bg-white/70 shadow-inner'
                  : 'border-purple-300/80 hover:border-purple-500 bg-purple-50/50 cursor-pointer hover:bg-purple-100/40'
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="audio/*"
                onChange={(e) => {
                  if (e.target.files?.[0]) handleLoadFile(e.target.files[0]);
                  e.target.value = '';
                }}
                className="hidden"
              />

              {file ? (
                <div className="w-full space-y-4">
                  <div className="flex items-center justify-between p-3 rounded-xl bg-purple-50 border border-purple-100">
                    <div className="flex items-center gap-2.5 min-w-0">
                      <div className="w-9 h-9 rounded-xl bg-purple-600 flex items-center justify-center text-white shadow-sm flex-shrink-0">
                        <FileAudio size={18} />
                      </div>
                      <div className="min-w-0">
                        <p className="text-xs font-bold text-[#1E1238] truncate">{audioMeta?.name}</p>
                        <p className="text-[10px] text-[#5B4E75] font-mono">{formatTime(duration)} audio clip</p>
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleLoadFile(null);
                        clearAnalysis('audio');
                      }}
                      className="text-gray-400 hover:text-rose-600 transition-colors p-1"
                      title="Clear audio"
                    >
                      <X size={15} />
                    </button>
                  </div>

                  {/* Cyan / Purple Glowing Waveform (Reference image equalizer style) */}
                  <div className="relative py-2">
                    {isAnalyzing && <div className="scan-beam" />}

                    <div className="flex items-end justify-between gap-[3px] h-20 bg-white p-3 rounded-xl border border-purple-100 shadow-sm">
                      {waveformPeaks.map((peak, idx) => {
                        const progress = duration > 0 ? currentTime / duration : 0;
                        const barProgress = idx / waveformPeaks.length;
                        const isPast = barProgress <= progress;

                        return (
                          <motion.div
                            key={idx}
                            initial={{ scaleY: 0.1 }}
                            animate={{
                              scaleY: isPlaying
                                ? Math.max(0.15, peak * (0.6 + Math.sin(Date.now() / 180 + idx) * 0.4))
                                : peak,
                            }}
                            transition={{ duration: 0.15 }}
                            className={`flex-1 rounded-full transition-colors duration-150 ${
                              isPast
                                ? 'bg-gradient-to-t from-purple-600 via-indigo-500 to-cyan-400 shadow-[0_0_6px_rgba(6,182,212,0.4)]'
                                : 'bg-purple-200/70'
                            }`}
                            style={{ height: '100%', transformOrigin: 'bottom' }}
                          />
                        );
                      })}
                    </div>
                  </div>

                  {/* Scrubber */}
                  <div className="space-y-1">
                    <input
                      type="range"
                      min={0}
                      max={duration || 100}
                      step={0.1}
                      value={currentTime}
                      onChange={handleSeek}
                      className="w-full h-1.5 rounded-full cursor-pointer appearance-none bg-purple-200 [accent-color:#6D28D9]"
                    />
                    <div className="flex justify-between text-[11px] font-mono font-bold text-[#8F81A8]">
                      <span>{formatTime(currentTime)}</span>
                      <span>{formatTime(duration)}</span>
                    </div>
                  </div>

                  {/* Play / Pause Pill Controls */}
                  <div className="flex items-center justify-between pt-1">
                    <button
                      onClick={togglePlay}
                      className="w-11 h-11 rounded-full bg-gradient-to-r from-purple-700 to-indigo-600 text-white flex items-center justify-center shadow-md shadow-purple-900/25 hover:scale-105 active:scale-95 transition-all"
                    >
                      {isPlaying ? <Pause size={18} /> : <Play size={18} className="ml-0.5" />}
                    </button>

                    <div className="flex items-center gap-2 text-purple-900">
                      <button
                        onClick={() => {
                          if (audioElementRef.current) {
                            const newMuted = !isMuted;
                            audioElementRef.current.muted = newMuted;
                            setIsMuted(newMuted);
                          }
                        }}
                        className="hover:text-purple-700 transition-colors"
                      >
                        {isMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
                      </button>
                      <input
                        type="range"
                        min={0}
                        max={1}
                        step={0.05}
                        value={isMuted ? 0 : volume}
                        onChange={(e) => {
                          const val = Number(e.target.value);
                          setVolume(val);
                          setIsMuted(false);
                          if (audioElementRef.current) {
                            audioElementRef.current.volume = val;
                            audioElementRef.current.muted = false;
                          }
                        }}
                        className="w-20 h-1.5 rounded-full cursor-pointer appearance-none bg-purple-200 [accent-color:#6D28D9]"
                      />
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center p-6 text-center">
                  <div className="w-16 h-16 rounded-3xl bg-white border border-purple-200 shadow-md flex items-center justify-center mb-3 text-purple-600 hover:scale-105 transition-transform">
                    <Mic size={30} />
                  </div>
                  <p className="text-sm font-bold text-[#1E1238] mb-1">
                    Upload Audio Evidence
                  </p>
                  <p className="text-xs text-[#5B4E75] max-w-[240px] mb-4">
                    Supports WAV, MP3, FLAC for spectral voice biometrics analysis
                  </p>
                  <button
                    type="button"
                    onClick={(e) => {
                      e.stopPropagation();
                      fileInputRef.current?.click();
                    }}
                    className="btn-ghost py-1.5 px-4 text-xs font-bold"
                  >
                    <FolderOpen size={13} /> Browse Audio
                  </button>
                </div>
              )}
            </div>

            {file && (
              <div className="flex gap-2 pt-1">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className="btn-ghost flex-1 py-2 text-xs font-bold"
                >
                  <RefreshCw size={12} /> Replace Audio File
                </button>
              </div>
            )}
          </div>

          {/* Action Trigger Card */}
          <div className="card space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Activity size={15} className="text-purple-700" />
                <span className="label-tag">Diagnostic Parameters</span>
              </div>
              <span className="text-[10px] font-mono font-bold text-cyan-700 bg-cyan-100 px-2 py-0.5 rounded-full">
                VOCODER SPECTRAL
              </span>
            </div>

            <div className="p-3.5 rounded-2xl bg-white border border-purple-100 space-y-2 text-xs text-[#5B4E75] shadow-sm">
              <div className="flex items-center justify-between">
                <span>Analysis Engine:</span>
                <span className="font-bold text-[#1E1238]">Voice Clone & TTS Inspector</span>
              </div>
              <div className="flex items-center justify-between">
                <span>FFT Resolution:</span>
                <span className="font-mono font-bold text-purple-700">High-Density Mel Bands</span>
              </div>
              <div className="flex items-center justify-between">
                <span>HiFi-GAN Filter:</span>
                <span className="text-emerald-600 font-bold">Enabled</span>
              </div>
            </div>

            {isAnalyzing ? (
              <div className="space-y-3 pt-1">
                <div className="flex gap-2">
                  <button disabled className="btn-primary flex-1 py-3 text-xs sm:text-sm font-bold">
                    <span className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Analyzing Voice Biometrics...
                  </button>
                  <button onClick={() => setConfirmCancel(true)} className="btn-danger py-3 px-4 text-xs font-bold">
                    <X size={15} /> Cancel
                  </button>
                </div>
                <div className="p-3.5 rounded-2xl bg-white border border-purple-200 space-y-2 shadow-sm">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-purple-700 flex items-center gap-1.5 font-bold">
                      <Sparkles size={13} className="animate-spin text-purple-600" />
                      {AUDIO_SCAN_STEPS[scanStepIndex]}
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
                <Volume2 size={18} />
                Run Acoustic Authenticity Scan
              </button>
            )}
          </div>
        </div>

        {/* Right Column: Results Station (7 cols) */}
        <div className="lg:col-span-7 space-y-5">
          {error ? (
            <div className="card p-6 border-rose-200 bg-rose-50/70 space-y-3">
              <div className="flex items-center gap-2 text-rose-700 font-bold">
                <ShieldAlert size={20} />
                <h3>Audio Analysis Error</h3>
              </div>
              <p className="text-xs sm:text-sm text-rose-900 leading-relaxed">{error}</p>
              <button onClick={handleAnalyze} className="btn-primary py-2 px-4 text-xs font-bold mt-2">
                Retry Scan
              </button>
            </div>
          ) : !results ? (
            /* Reference Styled Cube + Equalizer Empty State */
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
                    {isAnalyzing ? 'Decoding Acoustic Signal…' : 'Acoustic Biometric Station Ready'}
                  </h3>
                  <p className="text-xs sm:text-sm text-[#5B4E75] mt-1.5 leading-relaxed">
                    Upload an audio recording to identify AI voice cloning (ElevenLabs, Bark, Tortoise), speech synthesis, and vocoder artifacts.
                  </p>
                </div>

                <div className="grid grid-cols-2 gap-3 pt-2 text-left">
                  <div className="p-3.5 rounded-2xl bg-white border border-purple-100 shadow-sm">
                    <div className="text-xs font-bold text-purple-700 mb-1 flex items-center gap-1">
                      <AudioLines size={13} /> Vocoder Fingerprinting
                    </div>
                    <p className="text-[11px] text-[#5B4E75]">
                      Detects synthetic neural vocoder phase glitches and anomalies.
                    </p>
                  </div>
                  <div className="p-3.5 rounded-2xl bg-white border border-purple-100 shadow-sm">
                    <div className="text-xs font-bold text-cyan-600 mb-1 flex items-center gap-1">
                      <Activity size={13} /> Natural Biometrics
                    </div>
                    <p className="text-[11px] text-[#5B4E75]">
                      Evaluates natural breathing pauses, vocal formant dynamics, and pitch continuity.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            /* Results Hub */
            <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-5">
              {/* Verdict Card */}
              <div
                className={`card p-6 border relative overflow-hidden ${
                  results.risk_percent > 50
                    ? 'border-rose-300 bg-gradient-to-br from-rose-50 via-white to-rose-50/40 shadow-lg shadow-rose-500/10'
                    : 'border-emerald-300 bg-gradient-to-br from-emerald-50 via-white to-emerald-50/40 shadow-lg shadow-emerald-500/10'
                }`}
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="label-tag text-[10px]">Acoustic Classification</span>
                    <span
                      className="text-[10px] font-mono px-2.5 py-0.5 rounded-full font-bold uppercase tracking-wider"
                      style={{
                        color: getRiskColorRaw(results.risk_percent || 0),
                        backgroundColor: `${getRiskColorRaw(results.risk_percent || 0)}15`,
                        border: `1px solid ${getRiskColorRaw(results.risk_percent || 0)}30`,
                      }}
                    >
                      {getRiskLevel(results.risk_percent || 0)}
                    </span>
                  </div>
                  <h2 className="text-2xl sm:text-3xl font-black font-display tracking-tight text-[#1E1238] flex items-center gap-2.5">
                    {results.risk_percent > 50 ? (
                      <ShieldAlert className="text-rose-600 flex-shrink-0" size={30} />
                    ) : (
                      <ShieldCheck className="text-emerald-600 flex-shrink-0" size={30} />
                    )}
                    {results.verdict || 'Acoustic Scan Complete'}
                  </h2>
                  <p className="text-xs sm:text-sm text-[#5B4E75] leading-relaxed max-w-xl">
                    {results.explanation || 'Spectral analysis completed across frequency and phoneme distribution domains.'}
                  </p>
                </div>

                {results.verdict === 'AI-GENERATED' && (
                  <button
                    onClick={() => setComplaintOpen(true)}
                    className="btn-danger w-full py-2.5 text-xs sm:text-sm font-bold mt-4"
                  >
                    <ShieldAlert size={15} />
                    Raise Cyber Crime Complaint
                  </button>
                )}
              </div>

              {/* Compliance & cybercrime advisories (India IT Rules 2026) */}
              <CybercrimeRiskAdvisory risk={results.cybercrime_risk} />
              <ComplianceLabelBadge label={results.compliance_label} />

              {/* Dual Speedometer Dials */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="card p-5 flex flex-col items-center justify-between text-center">
                  <div className="flex items-center justify-between w-full mb-3 pb-2 border-b border-purple-100">
                    <span className="text-xs font-bold text-[#1E1238] uppercase tracking-wider">
                      Synthetic Voice Risk
                    </span>
                    <span className="text-[10px] font-mono font-bold text-rose-700 bg-rose-100 px-2.5 py-0.5 rounded-full">
                      AI PROBABILITY
                    </span>
                  </div>

                  <RiskGauge
                    percentage={results.risk_percent || 0}
                    label="AI RISK"
                    size={160}
                    strokeWidth={10}
                    showBadge={true}
                  />

                  <p className="text-xs text-[#5B4E75] mt-3 font-medium text-center leading-relaxed">
                    Likelihood of AI voice synthesis (ElevenLabs / Bark / Tortoise)
                  </p>
                </div>

                <div className="card p-5 flex flex-col items-center justify-between text-center">
                  <div className="flex items-center justify-between w-full mb-3 pb-2 border-b border-purple-100">
                    <span className="text-xs font-bold text-[#1E1238] uppercase tracking-wider">
                      Human Biometrics
                    </span>
                    <span className="text-[10px] font-mono font-bold text-emerald-700 bg-emerald-100 px-2.5 py-0.5 rounded-full">
                      AUTHENTICITY
                    </span>
                  </div>

                  <RiskGauge
                    percentage={results.authenticity_percentage || (100 - (results.risk_percent || 0))}
                    label="NATURAL"
                    size={160}
                    strokeWidth={10}
                    showBadge={true}
                  />

                  <p className="text-xs text-[#5B4E75] mt-3 font-medium text-center leading-relaxed">
                    Natural vocal tract acoustic and breathing coherence
                  </p>
                </div>
              </div>

              {/* Manipulation Profile & Audio Metrics */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                <div className="card p-4">
                  <span className="label-tag text-[10px] block mb-1">Detected Profile</span>
                  <p className="text-sm font-bold text-[#1E1238]">
                    {results.manipulation_type || 'Natural Human Speech'}
                  </p>
                </div>
                <div className="card p-4">
                  <span className="label-tag text-[10px] block mb-1">Duration</span>
                  <p className="text-sm font-bold font-mono text-purple-700">
                    {results.duration_sec ? `${results.duration_sec.toFixed(1)}s` : `${formatTime(duration)}`}
                  </p>
                </div>
                <div className="card p-4">
                  <span className="label-tag text-[10px] block mb-1">Segments Inspected</span>
                  <p className="text-sm font-bold font-mono text-[#1E1238]">
                    {results.segments_analyzed || Math.ceil(duration / 2) || 1}
                  </p>
                </div>
              </div>

              {/* Explanation Note */}
              {results.explanation && (
                <div className="card p-5 space-y-2">
                  <div className="flex items-center gap-2">
                    <Activity size={14} className="text-purple-700" />
                    <span className="label-tag">Spectral Findings & Biomarkers</span>
                  </div>
                  <p className="text-xs font-mono leading-relaxed text-[#5B4E75] p-3.5 rounded-xl bg-purple-50/70 border border-purple-100 whitespace-pre-wrap">
                    {results.explanation}
                  </p>
                </div>
              )}
            </motion.div>
          )}
        </div>
      </div>

      <ConfirmDialog
        open={confirmCancel}
        title="Cancel Voice Forensics"
        message="Are you sure you want to stop the active audio analysis?"
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
