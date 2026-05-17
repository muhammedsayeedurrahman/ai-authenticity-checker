import React from 'react';
import { motion } from 'framer-motion';
import { Shield, Upload, Cpu, CheckCircle, ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const fadeUp = {
  hidden: { opacity: 0, y: 14 },
  visible: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.1, duration: 0.45, ease: [0.22, 1, 0.36, 1] },
  }),
};

const STEPS = [
  { icon: Upload, title: 'Upload Media', desc: 'Image, video, or audio file' },
  { icon: Cpu, title: 'AI Analysis', desc: 'Multi-model forensic ensemble' },
  { icon: CheckCircle, title: 'Get Verdict', desc: 'Risk score & detailed report' },
];

export default function EmptyDashboard() {
  const navigate = useNavigate();

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      className="flex flex-col items-center justify-center py-12 px-4"
    >
      {/* Pulsing shield icon */}
      <motion.div variants={fadeUp} custom={0} className="relative mb-6">
        <div
          className="absolute inset-0 rounded-full"
          style={{
            background: 'radial-gradient(circle, rgba(59,130,246,0.25) 0%, transparent 70%)',
            animation: 'pulseGlow 3s ease-in-out infinite',
          }}
        />
        <div
          className="relative w-20 h-20 rounded-2xl flex items-center justify-center"
          style={{
            background: 'var(--accent-dim)',
            border: '1px solid rgba(59,130,246,0.25)',
            boxShadow: '0 0 30px rgba(59,130,246,0.15)',
          }}
        >
          <Shield size={36} style={{ color: 'var(--accent)' }} />
        </div>
      </motion.div>

      {/* Heading */}
      <motion.h2
        variants={fadeUp}
        custom={1}
        className="font-display text-2xl font-bold gradient-text mb-1"
      >
        System Ready
      </motion.h2>
      <motion.p
        variants={fadeUp}
        custom={1.5}
        className="text-[12px] mb-10"
        style={{ color: 'var(--text-3)' }}
      >
        No scans yet. Start your first forensic analysis in three steps.
      </motion.p>

      {/* Step cards */}
      <motion.div
        variants={fadeUp}
        custom={2}
        className="grid grid-cols-1 sm:grid-cols-3 gap-4 w-full max-w-2xl mb-10"
      >
        {STEPS.map((step, i) => {
          const Icon = step.icon;
          return (
            <div
              key={step.title}
              className="card p-5 flex flex-col items-center text-center"
            >
              <div className="flex items-center gap-2 mb-3">
                <span
                  className="w-5 h-5 rounded-full flex items-center justify-center text-[10px] font-bold"
                  style={{
                    background: 'var(--accent-dim)',
                    color: 'var(--accent)',
                    border: '1px solid rgba(59,130,246,0.2)',
                  }}
                >
                  {i + 1}
                </span>
                <Icon size={15} style={{ color: 'var(--accent)' }} />
              </div>
              <p className="text-[13px] font-semibold" style={{ color: 'var(--text-1)' }}>
                {step.title}
              </p>
              <p className="text-[11px] mt-0.5" style={{ color: 'var(--text-3)' }}>
                {step.desc}
              </p>
            </div>
          );
        })}
      </motion.div>

      {/* CTA */}
      <motion.button
        variants={fadeUp}
        custom={3}
        onClick={() => navigate('/image')}
        className="btn-primary px-8 py-3 text-[13px]"
      >
        Start Your First Scan
        <ArrowRight size={15} />
      </motion.button>
    </motion.div>
  );
}
