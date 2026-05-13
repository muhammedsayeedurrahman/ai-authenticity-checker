import React from 'react';
import { motion } from 'framer-motion';
import { Server, CheckCircle, XCircle, Cpu, Zap, Layers, Info, RefreshCw } from 'lucide-react';
import useForensicStore from '../store/useForensicStore';

export default function Settings() {
  const { systemStatus, isStatusLoading, statusError, fetchStatus } = useForensicStore();

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <header className="mb-8 flex justify-between items-end">
        <div>
          <h2 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <Server className="text-accent" /> Settings & Status
          </h2>
          <p className="text-text-muted mt-2">System configuration and model infrastructure monitor.</p>
        </div>
        <button
          onClick={fetchStatus}
          disabled={isStatusLoading}
          className="px-4 py-2 border border-border-hover rounded-lg text-accent hover:bg-accent/5 text-sm font-medium transition-all flex items-center gap-2"
        >
          <RefreshCw size={14} className={isStatusLoading ? 'animate-spin' : ''} />
          {isStatusLoading ? "Syncing..." : "Refresh"}
        </button>
      </header>

      {statusError && (
        <div className="p-4 bg-accent-danger/10 border border-accent-danger rounded-xl text-accent-danger">
          System Offline or Unreachable: {statusError}
        </div>
      )}

      {/* Model Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-card p-6">
          <h3 className="text-sm font-bold text-text-secondary uppercase tracking-widest mb-6 border-b border-border-subtle pb-4">
            Loaded AI Models
          </h3>
          <div className="space-y-3">
            {(systemStatus?.loaded_models?.length || 0) === 0 ? (
              <p className="text-text-muted text-sm italic">No active models detected.</p>
            ) : (
              systemStatus.loaded_models.map((model) => (
                <div key={model} className="flex justify-between items-center bg-accent-success/5 border border-accent-success/20 p-3 rounded-lg">
                  <span className="text-sm font-medium text-text-primary">{model}</span>
                  <CheckCircle size={18} className="text-accent-success" />
                </div>
              ))
            )}
          </div>
        </div>

        <div className="glass-card p-6">
          <h3 className="text-sm font-bold text-accent-danger/80 uppercase tracking-widest mb-6 border-b border-border-subtle pb-4">
            Missing / Unloaded Models
          </h3>
          <div className="space-y-3">
            {(systemStatus?.missing_models?.length || 0) === 0 ? (
              <p className="text-text-muted text-sm italic">All requested models are operational.</p>
            ) : (
              systemStatus.missing_models.map((model) => (
                <div key={model} className="flex justify-between items-center bg-accent-danger/5 border border-accent-danger/20 p-3 rounded-lg">
                  <span className="text-sm font-medium text-text-muted">{model}</span>
                  <XCircle size={18} className="text-accent-danger opacity-50" />
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* System Info Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="glass-card p-6 flex flex-col justify-center items-center text-center">
          <Cpu size={40} className="mb-4 text-accent" />
          <h4 className="text-xs text-text-secondary font-bold uppercase tracking-widest mb-2">Compute Device</h4>
          <p className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-accent to-indigo-400 uppercase">
            {systemStatus?.device || "UNKNOWN"}
          </p>
        </div>

        <div className="glass-card p-6 flex flex-col justify-center items-center text-center">
          <Zap size={40} className="mb-4 text-accent-warning" />
          <h4 className="text-xs text-text-secondary font-bold uppercase tracking-widest mb-2">Fast Mode (CorefakeNet)</h4>
          <div className={`px-4 py-1 rounded-full text-xs font-bold ${
            systemStatus?.corefakenet_available
              ? 'bg-accent-success/10 text-accent-success border border-accent-success/30'
              : 'bg-accent-danger/10 text-accent-danger border border-accent-danger/30'
          }`}>
            {systemStatus?.corefakenet_available ? "ONLINE" : "OFFLINE"}
          </div>
        </div>

        <div className="glass-card p-6 flex flex-col justify-center items-center text-center">
          <Layers size={40} className="mb-4 text-accent" />
          <h4 className="text-xs text-text-secondary font-bold uppercase tracking-widest mb-2">Fusion Engine</h4>
          <div className={`px-4 py-1 rounded-full text-xs font-bold ${
            systemStatus?.fusion_mlp_available
              ? 'bg-accent-success/10 text-accent-success border border-accent-success/30'
              : 'bg-accent-danger/10 text-accent-danger border border-accent-danger/30'
          }`}>
            {systemStatus?.fusion_mlp_available ? "ONLINE" : "OFFLINE / HEURISTIC"}
          </div>
        </div>
      </div>

      {/* API Info */}
      <div className="glass-card p-6">
        <h3 className="text-sm font-bold text-text-secondary uppercase tracking-widest mb-4 flex items-center gap-2">
          <Info size={16} /> API Configuration
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div className="flex justify-between py-2 border-b border-border-subtle">
            <span className="text-text-muted">API Base URL</span>
            <code className="text-accent font-mono text-xs">/api/v1</code>
          </div>
          <div className="flex justify-between py-2 border-b border-border-subtle">
            <span className="text-text-muted">Image Rate Limit</span>
            <span className="text-text-primary">30/min</span>
          </div>
          <div className="flex justify-between py-2 border-b border-border-subtle">
            <span className="text-text-muted">Video Rate Limit</span>
            <span className="text-text-primary">10/min</span>
          </div>
          <div className="flex justify-between py-2 border-b border-border-subtle">
            <span className="text-text-muted">Audio Rate Limit</span>
            <span className="text-text-primary">20/min</span>
          </div>
          <div className="flex justify-between py-2 border-b border-border-subtle">
            <span className="text-text-muted">Version</span>
            <span className="text-text-primary">2.0.0</span>
          </div>
          <div className="flex justify-between py-2 border-b border-border-subtle">
            <span className="text-text-muted">Total Models</span>
            <span className="text-text-primary">{systemStatus?.total || 0}</span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}
