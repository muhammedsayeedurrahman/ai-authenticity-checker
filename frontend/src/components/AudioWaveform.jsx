import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { AudioLines } from 'lucide-react';

const NUM_BARS = 96;

/**
 * Real waveform decoded client-side from the actual uploaded audio file
 * (Web Audio API — no backend involvement). Static, one-time render with a
 * build-in reveal; no playback controls.
 */
export default function AudioWaveform({ file }) {
  const [bars, setBars] = useState([]);
  const [status, setStatus] = useState('idle'); // idle | decoding | ready | error

  useEffect(() => {
    if (!file) {
      setBars([]);
      setStatus('idle');
      return;
    }

    let cancelled = false;
    setStatus('decoding');
    setBars([]);

    (async () => {
      let audioCtx;
      try {
        const arrayBuffer = await file.arrayBuffer();
        const AudioContextCls = window.AudioContext || window.webkitAudioContext;
        audioCtx = new AudioContextCls();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);
        if (cancelled) return;

        const raw = audioBuffer.getChannelData(0);
        const blockSize = Math.max(1, Math.floor(raw.length / NUM_BARS));
        const peaks = [];
        for (let i = 0; i < NUM_BARS; i++) {
          const start = i * blockSize;
          let max = 0;
          for (let j = 0; j < blockSize && start + j < raw.length; j++) {
            const abs = Math.abs(raw[start + j]);
            if (abs > max) max = abs;
          }
          peaks.push(max);
        }
        const peakMax = Math.max(...peaks, 0.0001);
        const normalized = peaks.map((p) => Math.max(0.04, p / peakMax));

        if (!cancelled) {
          setBars(normalized);
          setStatus('ready');
        }
      } catch {
        if (!cancelled) setStatus('error');
      } finally {
        if (audioCtx) {
          try { await audioCtx.close(); } catch { /* already closed */ }
        }
      }
    })();

    return () => { cancelled = true; };
  }, [file]);

  if (status === 'idle') return null;

  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-3">
        <AudioLines size={13} className="text-accent" />
        <span className="label-tag">Waveform</span>
      </div>

      {status === 'error' ? (
        <p className="text-sm text-text-3 text-center py-6">Waveform preview unavailable</p>
      ) : status === 'decoding' ? (
        <div className="flex items-end gap-[2px] h-20" aria-hidden="true">
          {Array.from({ length: NUM_BARS }).map((_, i) => (
            <div key={i} className="flex-1 bg-white/[0.05] rounded-sm" style={{ height: '30%' }} />
          ))}
        </div>
      ) : (
        <div className="flex items-end gap-[2px] h-20">
          {bars.map((h, i) => (
            <motion.div
              key={i}
              className="flex-1 rounded-sm bg-accent"
              initial={{ scaleY: 0 }}
              animate={{ scaleY: h }}
              transition={{ duration: 0.4, delay: Math.min(i * 0.004, 0.3), ease: [0.22, 1, 0.36, 1] }}
              style={{ height: '100%', transformOrigin: 'bottom' }}
            />
          ))}
        </div>
      )}
    </div>
  );
}
