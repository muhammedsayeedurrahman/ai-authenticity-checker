import React from 'react';
import { motion } from 'framer-motion';
import { Server, CheckCircle, XCircle, Cpu, Zap } from 'lucide-react';
import useForensicStore from '../store/useForensicStore';

export default function StatusPage() {
  const { systemStatus, isStatusLoading, statusError, fetchStatus } = useForensicStore();

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <header className="mb-8 flex justify-between items-end">
        <div>
          <h2 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <Server className="text-accent-cyan" /> Backend Infrastructure
          </h2>
          <p className="text-text-muted mt-2">Neural network models and hardware status monitor.</p>
        </div>
        <button 
          onClick={fetchStatus}
          disabled={isStatusLoading}
          className="px-4 py-2 border border-border-glow rounded-lg text-accent-cyan hover:bg-[rgba(0,240,255,0.05)] text-sm font-medium transition-all"
        >
          {isStatusLoading ? "Syncing..." : "Refresh Status"}
        </button>
      </header>

      {statusError && (
        <div className="p-4 bg-[rgba(236,72,153,0.1)] border border-accent-pink rounded-xl text-accent-pink">
           System Offline or Unreachable: {statusError}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-card p-6">
          <h3 className="text-sm font-bold text-text-secondary uppercase tracking-widest mb-6 border-b border-border-subtle pb-4">
            Loaded AI Models
          </h3>
          
          <div className="space-y-4">
            {systemStatus?.loaded_models?.length === 0 ? (
              <p className="text-text-muted text-sm italic">No active models detected.</p>
            ) : (
              systemStatus?.loaded_models?.map(model => (
                <div key={model} className="flex justify-between items-center bg-[rgba(16,185,129,0.05)] border border-[rgba(16,185,129,0.2)] p-3 rounded-lg">
                  <span className="text-sm font-medium text-text-primary">{model}</span>
                  <CheckCircle size={18} className="text-accent-green" />
                </div>
              ))
            )}
          </div>
        </div>

        <div className="glass-card p-6">
          <h3 className="text-sm font-bold text-[rgba(236,72,153,0.8)] uppercase tracking-widest mb-6 border-b border-border-subtle pb-4">
            Missing / Unloaded Models
          </h3>
          
          <div className="space-y-4">
             {systemStatus?.missing_models?.length === 0 ? (
              <p className="text-text-muted text-sm italic">All requested models are operational.</p>
            ) : (
              systemStatus?.missing_models?.map(model => (
                <div key={model} className="flex justify-between items-center bg-[rgba(236,72,153,0.05)] border border-[rgba(236,72,153,0.2)] p-3 rounded-lg">
                  <span className="text-sm font-medium text-text-muted">{model}</span>
                  <XCircle size={18} className="text-accent-pink opacity-50" />
                </div>
              ))
            )}
          </div>
        </div>

        <div className="md:col-span-2 grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="glass-card p-6 flex flex-col justify-center items-center text-center">
            <Cpu size={40} className="mb-4 text-accent-violet" />
            <h4 className="text-xs text-text-secondary font-bold uppercase tracking-widest mb-2">Compute Device</h4>
            <p className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-accent-cyan to-accent-violet uppercase">
              {systemStatus?.device || "UNKNOWN"}
            </p>
          </div>
          
          <div className="glass-card p-6 flex flex-col justify-center items-center text-center">
            <Zap size={40} className="mb-4 text-[#F59E0B]" />
            <h4 className="text-xs text-text-secondary font-bold uppercase tracking-widest mb-2">Fast Mode (Corefakenet)</h4>
            <div className={`px-4 py-1 rounded-full text-xs font-bold ${
              systemStatus?.corefakenet_ready || systemStatus?.corefakenet_available
                ? 'bg-[rgba(16,185,129,0.1)] text-accent-green border border-[rgba(16,185,129,0.3)]'
                : 'bg-[rgba(236,72,153,0.1)] text-accent-pink border border-[rgba(236,72,153,0.3)]'
            }`}>
              {systemStatus?.corefakenet_ready || systemStatus?.corefakenet_available ? "ONLINE" : "OFFLINE"}
            </div>
          </div>
          
          <div className="glass-card p-6 flex flex-col justify-center items-center text-center">
            <Layers size={40} className="mb-4 text-accent-cyan" />
            <h4 className="text-xs text-text-secondary font-bold uppercase tracking-widest mb-2">Fusion Engine</h4>
            <div className={`px-4 py-1 rounded-full text-xs font-bold ${
              systemStatus?.fusion_mlp_available
                ? 'bg-[rgba(16,185,129,0.1)] text-accent-green border border-[rgba(16,185,129,0.3)]'
                : 'bg-[rgba(236,72,153,0.1)] text-accent-pink border border-[rgba(236,72,153,0.3)]'
            }`}>
              {systemStatus?.fusion_mlp_available ? "ONLINE" : "OFFLINE / HEURISTIC"}
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
