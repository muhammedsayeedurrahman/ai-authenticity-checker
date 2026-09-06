import React from 'react';
import { motion } from 'framer-motion';
import { Image, Film, Mic, Zap } from 'lucide-react';

const NODES = [
  { key: 'image', Icon: Image, x: 15 },
  { key: 'video', Icon: Film, x: 50 },
  { key: 'audio', Icon: Mic, x: 85 },
];

const NODE_Y = 18;
const HUB_Y = 82;
const ACCENT = '#6D28D9';

/**
 * Connects the 3 modality upload slots to a central "fusion" node —
 * lines/nodes light up as each modality is provided, connectors pulse
 * while analysis is running. Purely presentational (SVG lines + HTML
 * icon badges positioned on matching percentage coordinates).
 */
export default function FusionVisualizer({ image, video, audio, isAnalyzing }) {
  const active = { image: !!image, video: !!video, audio: !!audio };
  const anyActive = active.image || active.video || active.audio;

  return (
    <div className="relative h-28" aria-hidden="true">
      <svg
        className="absolute inset-0 w-full h-full"
        viewBox="0 0 100 100"
        preserveAspectRatio="none"
      >
        {NODES.map(({ key, x }) => (
          <motion.line
            key={key}
            x1={x} y1={NODE_Y} x2={50} y2={HUB_Y}
            stroke={active[key] ? ACCENT : 'rgba(139, 92, 246, 0.15)'}
            strokeWidth={0.6}
            strokeDasharray={active[key] ? '4 3' : undefined}
            vectorEffect="non-scaling-stroke"
            animate={
              isAnalyzing && active[key]
                ? { strokeDashoffset: [0, -14] }
                : { strokeDashoffset: 0 }
            }
            transition={{ duration: 0.6, repeat: isAnalyzing ? Infinity : 0, ease: 'linear' }}
          />
        ))}
      </svg>

      {NODES.map(({ key, Icon, x }) => (
        <div
          key={key}
          className="absolute -translate-x-1/2 -translate-y-1/2 flex flex-col items-center"
          style={{ left: `${x}%`, top: `${NODE_Y}%` }}
        >
          <motion.div
            animate={{ opacity: active[key] ? 1 : 0.4, scale: active[key] ? 1.05 : 1 }}
            transition={{ duration: 0.2 }}
            className={`w-9 h-9 rounded-xl flex items-center justify-center border ${
              active[key]
                ? 'bg-purple-100 border-purple-400 text-purple-700 shadow-sm'
                : 'bg-white/60 border-purple-200/60 text-[#8F81A8]'
            }`}
          >
            <Icon size={16} />
          </motion.div>
        </div>
      ))}

      <div
        className="absolute -translate-x-1/2 -translate-y-1/2 flex flex-col items-center"
        style={{ left: '50%', top: `${HUB_Y}%` }}
      >
        <motion.div
          animate={{
            opacity: anyActive ? 1 : 0.4,
            scale: anyActive ? (isAnalyzing ? [1, 1.12, 1] : 1) : 0.9,
          }}
          transition={
            isAnalyzing
              ? { duration: 1.1, repeat: Infinity, ease: 'easeInOut' }
              : { duration: 0.2 }
          }
          className={`w-10 h-10 rounded-full flex items-center justify-center border ${
            anyActive
              ? 'bg-gradient-to-tr from-purple-700 to-indigo-600 text-white border-purple-400 shadow-md shadow-purple-500/30'
              : 'bg-white/60 border-purple-200/60 text-[#8F81A8]'
          }`}
        >
          <Zap size={18} />
        </motion.div>
      </div>
    </div>
  );
}
