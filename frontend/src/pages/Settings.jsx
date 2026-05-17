import React from 'react';
import { motion } from 'framer-motion';
import { CheckCircle, XCircle, Cpu, Zap, Layers, Info, RefreshCw } from 'lucide-react';
import useForensicStore from '../store/useForensicStore';

const fadeUp = {
  hidden:  { opacity: 0, y: 14 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.4, ease: [0.22, 1, 0.36, 1] } },
};

export default function Settings() {
  const { systemStatus, isStatusLoading, statusError, fetchStatus } = useForensicStore();

  return (
    <motion.div initial="hidden" animate="visible" variants={fadeUp} className="space-y-6">
      {/* Header */}
      <header className="flex justify-between items-center">
        <h1 className="font-display text-xl font-bold" style={{ color: 'var(--text-1)' }}>System Status</h1>
        <button
          onClick={fetchStatus}
          disabled={isStatusLoading}
          className="btn-ghost flex items-center gap-2 text-[12px]"
        >
          <RefreshCw size={14} className={isStatusLoading ? 'animate-spin' : ''} />
          {isStatusLoading ? 'Syncing...' : 'Refresh'}
        </button>
      </header>

      {statusError && (
        <div
          className="p-3 rounded-lg text-[12px]"
          style={{ background: 'rgba(251,113,133,0.08)', border: '1px solid rgba(251,113,133,0.20)', color: 'var(--risk-critical)' }}
        >
          System Offline or Unreachable: {statusError}
        </div>
      )}

      {/* Model Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="card">
          <p className="text-[10px] font-bold uppercase tracking-widest mb-4" style={{ color: 'var(--text-3)' }}>
            Loaded AI Models
          </p>
          <div className="space-y-2">
            {(systemStatus?.loaded_models?.length || 0) === 0 ? (
              <p className="text-[12px] italic" style={{ color: 'var(--text-2)' }}>No active models detected.</p>
            ) : (
              systemStatus.loaded_models.map((model) => (
                <div
                  key={model}
                  className="flex justify-between items-center p-2.5 rounded-lg"
                  style={{ background: 'rgba(52,211,153,0.05)', border: '1px solid rgba(52,211,153,0.15)', boxShadow: '0 0 8px rgba(52,211,153,0.08)' }}
                >
                  <span className="text-[12px] font-medium" style={{ color: 'var(--text-1)' }}>{model}</span>
                  <CheckCircle size={16} style={{ color: 'var(--risk-clear)' }} />
                </div>
              ))
            )}
          </div>
        </div>

        <div className="card">
          <p className="text-[10px] font-bold uppercase tracking-widest mb-4" style={{ color: 'var(--text-3)' }}>
            Missing / Unloaded Models
          </p>
          <div className="space-y-2">
            {(systemStatus?.missing_models?.length || 0) === 0 ? (
              <p className="text-[12px] italic" style={{ color: 'var(--text-2)' }}>All requested models are operational.</p>
            ) : (
              systemStatus.missing_models.map((model) => (
                <div
                  key={model}
                  className="flex justify-between items-center p-2.5 rounded-lg"
                  style={{ background: 'rgba(251,113,133,0.05)', border: '1px solid rgba(251,113,133,0.15)', boxShadow: '0 0 8px rgba(251,113,133,0.08)' }}
                >
                  <span className="text-[12px] font-medium" style={{ color: 'var(--text-2)' }}>{model}</span>
                  <XCircle size={16} style={{ color: 'var(--risk-critical)' }} />
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* System Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        <div className="card">
          <div className="flex items-center gap-2 mb-2">
            <Cpu size={16} style={{ color: 'var(--accent)' }} />
            <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: 'var(--text-3)' }}>Compute Device</span>
          </div>
          <p className="text-lg font-semibold uppercase" style={{ color: 'var(--text-1)' }}>
            {systemStatus?.device || 'UNKNOWN'}
          </p>
        </div>

        <div className="card">
          <div className="flex items-center gap-2 mb-2">
            <Zap size={16} style={{ color: 'var(--accent)' }} />
            <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: 'var(--text-3)' }}>Fast Mode</span>
          </div>
          <span
            className="inline-block px-3 py-1 rounded text-[11px] font-bold"
            style={{
              background: systemStatus?.corefakenet_available ? 'rgba(52,211,153,0.10)' : 'rgba(251,113,133,0.10)',
              color: systemStatus?.corefakenet_available ? 'var(--risk-clear)' : 'var(--risk-critical)',
              border: `1px solid ${systemStatus?.corefakenet_available ? 'rgba(52,211,153,0.25)' : 'rgba(251,113,133,0.25)'}`,
            }}
          >
            {systemStatus?.corefakenet_available ? 'ONLINE' : 'OFFLINE'}
          </span>
        </div>

        <div className="card">
          <div className="flex items-center gap-2 mb-2">
            <Layers size={16} style={{ color: 'var(--accent)' }} />
            <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: 'var(--text-3)' }}>Fusion Engine</span>
          </div>
          <span
            className="inline-block px-3 py-1 rounded text-[11px] font-bold"
            style={{
              background: systemStatus?.fusion_mlp_available ? 'rgba(52,211,153,0.10)' : 'rgba(251,113,133,0.10)',
              color: systemStatus?.fusion_mlp_available ? 'var(--risk-clear)' : 'var(--risk-critical)',
              border: `1px solid ${systemStatus?.fusion_mlp_available ? 'rgba(52,211,153,0.25)' : 'rgba(251,113,133,0.25)'}`,
            }}
          >
            {systemStatus?.fusion_mlp_available ? 'ONLINE' : 'OFFLINE / HEURISTIC'}
          </span>
        </div>
      </div>

      {/* API Info */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <Info size={16} style={{ color: 'var(--text-3)' }} />
          <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: 'var(--text-3)' }}>API Configuration</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
          {[
            { label: 'API Base URL', value: '/api/v1', mono: true },
            { label: 'Image Rate Limit', value: '30/min' },
            { label: 'Video Rate Limit', value: '10/min' },
            { label: 'Audio Rate Limit', value: '20/min' },
            { label: 'Version', value: '2.0.0' },
            { label: 'Total Models', value: String(systemStatus?.total || 0) },
          ].map(({ label, value, mono }) => (
            <div
              key={label}
              className="flex justify-between py-2.5"
              style={{ borderBottom: '1px solid var(--border-dim)' }}
            >
              <span className="text-[12px]" style={{ color: 'var(--text-2)' }}>{label}</span>
              <span
                className={`text-[12px] font-medium ${mono ? 'font-mono' : ''}`}
                style={{ color: mono ? 'var(--accent)' : 'var(--text-1)' }}
              >
                {value}
              </span>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}
