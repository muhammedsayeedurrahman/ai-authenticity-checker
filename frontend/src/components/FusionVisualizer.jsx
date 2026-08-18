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
          <line
            key={key}
            x1={x} y1={NODE_Y} x2={50} y2={HUB_Y}
            stroke={active[key] ? '#3B82F6' : 'rgba(255,255,255,0.08)'}
            strokeWidth={0.6}
            strokeDasharray={active[key] ? '4 3' : undefined}
            className={isAnalyzing && active[key] ? 'fusion-line-pulse' : ''}
            vectorEffect="non-scaling-stroke"
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
            className={`w-9 h-9 rounded-lg flex items-center justify-center border ${
              active[key]
                ? 'bg-accent-dim border-accent/30 text-accent'
                : 'bg-bg-inset border-border-dim text-text-3'
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
          animate={{ opacity: anyActive ? 1 : 0.4, scale: anyActive ? 1 : 0.9 }}
          transition={{ duration: 0.2 }}
          className={`w-10 h-10 rounded-full flex items-center justify-center border ${
            anyActive
              ? 'bg-accent text-white border-accent shadow-glow-blue'
              : 'bg-bg-inset border-border-dim text-text-3'
          }`}
        >
          <Zap size={18} className={isAnalyzing ? 'animate-pulse' : ''} />
        </motion.div>
      </div>
    </div>
  );
}
