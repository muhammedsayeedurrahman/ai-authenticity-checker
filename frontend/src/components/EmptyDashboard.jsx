import React, { useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Shield, Upload, Cpu, CheckCircle, ArrowRight, Sparkles } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { staggerFadeUp } from '../utils/animations';
import { detectMediaRoute } from '../utils/format';
import useForensicStore from '../store/useForensicStore';

const STEPS = [
  { icon: Upload, title: '1. Select Media', desc: 'Drop image, voice recording, or video clip' },
  { icon: Cpu, title: '2. Multi-Model AI Scan', desc: '7 neural networks verify frequencies & pixels' },
  { icon: CheckCircle, title: '3. Instant Intelligence', desc: 'Risk dial, GradCAM heatmap & report' },
];

export default function EmptyDashboard() {
  const navigate = useNavigate();
  const { setPendingFile } = useForensicStore();
  const fileInputRef = useRef(null);

  const handleFile = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setPendingFile(file);
    navigate(detectMediaRoute(file));
  }, [navigate, setPendingFile]);

  return (
    <motion.div initial="hidden" animate="visible" className="flex flex-col items-center justify-center py-10 px-4">
      {/* 3D-styled Robot Orb */}
      <motion.div variants={staggerFadeUp} custom={0} className="mb-5">
        <div className="w-24 h-24 rounded-full bg-gradient-to-tr from-purple-700 via-indigo-600 to-pink-500 p-1 shadow-2xl shadow-purple-500/25 flex items-center justify-center">
          <div className="w-full h-full rounded-full bg-[#1E1238] flex items-center justify-center relative">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-pink-400 shadow-[0_0_10px_#F472B6] animate-pulse" />
              <div className="w-3 h-3 rounded-full bg-cyan-400 shadow-[0_0_10px_#38BDF8] animate-pulse" />
            </div>
          </div>
        </div>
      </motion.div>

      <motion.h2 variants={staggerFadeUp} custom={1} className="font-display text-3xl font-black text-[#1E1238] tracking-tight mb-2 text-center">
        Forensic Command Station Active
      </motion.h2>
      <motion.p variants={staggerFadeUp} custom={2} className="text-sm text-[#5B4E75] font-medium max-w-md text-center mb-8 leading-relaxed">
        Verify digital images, cloned voices, and deepfake videos with multi-spectral neural networks.
      </motion.p>

      {/* Step cards */}
      <motion.div variants={staggerFadeUp} custom={3} className="grid grid-cols-1 sm:grid-cols-3 gap-4 w-full max-w-3xl mb-8">
        {STEPS.map((step) => {
          const Icon = step.icon;
          return (
            <div key={step.title} className="card p-5 flex flex-col items-center text-center space-y-2">
              <div className="w-10 h-10 rounded-2xl bg-purple-100 flex items-center justify-center text-purple-700 border border-purple-200 shadow-sm">
                <Icon size={18} />
              </div>
              <p className="text-sm font-bold text-[#1E1238]">{step.title}</p>
              <p className="text-xs text-[#5B4E75] leading-relaxed">{step.desc}</p>
            </div>
          );
        })}
      </motion.div>

      <motion.div variants={staggerFadeUp} custom={4}>
        <input
          ref={fileInputRef}
          type="file"
          accept="image/*,video/*,audio/*"
          onChange={handleFile}
          className="hidden"
          aria-label="Upload media file"
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          className="btn-primary px-8 py-3.5 text-sm font-bold shadow-lg shadow-purple-900/20"
        >
          <Sparkles size={16} />
          Start First Forensic Scan
          <ArrowRight size={16} />
        </button>
      </motion.div>
    </motion.div>
  );
}
