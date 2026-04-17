import React from 'react';
import { motion } from 'framer-motion';
import { ShieldAlert, Image, Film, Mic, Layers } from 'lucide-react';
import { Link } from 'react-router-dom';

export default function Dashboard() {
  const cards = [
    {
      to: "/image",
      title: "Image Forensics",
      desc: "Detect manipulated pixels and synthetic faces",
      icon: <Image size={32} className="text-accent-cyan" />,
      bg: "from-[rgba(0,240,255,0.05)] to-transparent",
      border: "hover:border-accent-cyan"
    },
    {
      to: "/video",
      title: "Video Forensics",
      desc: "Frame-by-frame temporal consistency alignment",
      icon: <Film size={32} className="text-accent-violet" />,
      bg: "from-[rgba(168,85,247,0.05)] to-transparent",
      border: "hover:border-accent-violet"
    },
    {
      to: "/audio",
      title: "Audio Forensics",
      desc: "Spectrogram analysis for synthetic voice cloning",
      icon: <Mic size={32} className="text-accent-green" />,
      bg: "from-[rgba(16,185,129,0.05)] to-transparent",
      border: "hover:border-accent-green"
    },
    {
      to: "/multimodal",
      title: "Multimodal Fusion",
      desc: "Compound confidence matrix using all available factors",
      icon: <Layers size={32} className="text-[#F59E0B]" />,
      bg: "from-[rgba(245,158,11,0.05)] to-transparent",
      border: "hover:border-[#F59E0B]"
    }
  ];

  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="flex flex-col items-center justify-center min-h-[85vh] text-center space-y-12">
      <div className="space-y-6">
        <div className="w-24 h-24 mx-auto rounded-3xl bg-gradient-to-br from-accent-cyan to-accent-violet flex items-center justify-center shadow-[0_0_50px_rgba(0,240,255,0.3)]">
          <ShieldAlert className="text-white" size={48} />
        </div>
        <h2 className="text-5xl font-black tracking-tighter">
          Intelligence <span className="bg-clip-text text-transparent bg-gradient-to-r from-accent-cyan to-accent-violet">Command Center</span>
        </h2>
        <p className="text-text-secondary max-w-xl mx-auto text-lg">
          Select an analysis vector below to initialize the neural detection pipeline.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 w-full max-w-6xl">
        {cards.map((card) => (
          <Link 
            key={card.to} 
            to={card.to}
            className={`group relative glass-card p-6 border-border-subtle overflow-hidden transition-all duration-500 hover:scale-105 hover:-translate-y-2 ${card.border}`}
          >
            <div className={`absolute inset-0 bg-gradient-to-b ${card.bg} opacity-0 group-hover:opacity-100 transition-opacity duration-500`} />
            <div className="relative z-10 flex flex-col items-center text-center space-y-4">
              <div className="p-4 rounded-xl bg-background border border-[rgba(255,255,255,0.05)] group-hover:scale-110 transition-transform duration-500">
                {card.icon}
              </div>
              <div>
                <h3 className="font-bold text-white mb-2 tracking-wide">{card.title}</h3>
                <p className="text-xs text-text-muted">{card.desc}</p>
              </div>
            </div>
          </Link>
        ))}
      </div>
    </motion.div>
  );
}
