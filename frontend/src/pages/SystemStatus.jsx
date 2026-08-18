import React from 'react';
import { CheckCircle, XCircle, Cpu, Zap, Layers, Info, RefreshCw, Activity } from 'lucide-react';
import PageHeader from '../components/PageHeader';
import useForensicStore from '../store/useForensicStore';

export default function SystemStatus() {
  const { systemStatus, isStatusLoading, statusError, fetchStatus } = useForensicStore();

  return (
    <div className="space-y-6">
      <PageHeader
        icon={Activity}
        title="System Status"
        subtitle="ML pipeline health and configuration"
        actions={
          <button
            onClick={fetchStatus}
            disabled={isStatusLoading}
            className="btn-ghost flex items-center gap-2"
          >
            <RefreshCw size={14} className={isStatusLoading ? 'animate-spin' : ''} />
            {isStatusLoading ? 'Syncing...' : 'Refresh'}
          </button>
        }
      />

      {statusError && (
        <div
          role="alert"
          className="p-3 rounded-lg text-sm bg-risk-criticalDim text-risk-critical border border-[rgba(251,113,133,0.20)]"
        >
          System Offline or Unreachable: {statusError}
        </div>
      )}

      {/* Model Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        <div className="card">
          <p className="label-tag mb-4">Loaded AI Models</p>
          <div className="space-y-2">
            {(systemStatus?.loaded_models?.length || 0) === 0 ? (
              <p className="text-sm italic text-text-2">No active models detected.</p>
            ) : (
              systemStatus.loaded_models.map((model) => (
                <div
                  key={model}
                  className="flex justify-between items-center p-2.5 rounded-lg bg-risk-clearDim border border-[rgba(34,197,94,0.15)]"
                >
                  <span className="text-sm font-medium text-text-1">{model}</span>
                  <CheckCircle size={16} className="text-risk-clear" />
                </div>
              ))
            )}
          </div>
        </div>

        <div className="card">
          <p className="label-tag mb-4">Missing / Unloaded Models</p>
          <div className="space-y-2">
            {(systemStatus?.missing_models?.length || 0) === 0 ? (
              <p className="text-sm italic text-text-2">All models operational.</p>
            ) : (
              systemStatus.missing_models.map((model) => (
                <div
                  key={model}
                  className="flex justify-between items-center p-2.5 rounded-lg bg-risk-criticalDim border border-[rgba(251,113,133,0.15)]"
                >
                  <span className="text-sm font-medium text-text-2">{model}</span>
                  <XCircle size={16} className="text-risk-critical" />
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      {/* System Info */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {[
          { icon: Cpu, title: 'Compute Device', value: systemStatus?.device || 'UNKNOWN' },
          {
            icon: Zap, title: 'Fast Mode',
            value: systemStatus?.corefakenet_available ? 'ONLINE' : 'OFFLINE',
            ok: systemStatus?.corefakenet_available,
          },
          {
            icon: Layers, title: 'Fusion Engine',
            value: systemStatus?.fusion_mlp_available ? 'LEARNED' : 'HEURISTIC',
            ok: systemStatus?.fusion_mlp_available,
          },
        ].map(({ icon, title, value, ok }) => {
          const Icon = icon;
          return (
            <div key={title} className="card">
              <div className="flex items-center gap-2 mb-2">
                <Icon size={16} className="text-accent" />
                <span className="label-tag">{title}</span>
              </div>
              {ok !== undefined ? (
                <span
                  className={`inline-block px-3 py-1 rounded text-xs font-bold border ${
                    ok
                      ? 'bg-risk-clearDim text-risk-clear border-[rgba(52,211,153,0.25)]'
                      : 'bg-risk-criticalDim text-risk-critical border-[rgba(251,113,133,0.25)]'
                  }`}
                >
                  {value}
                </span>
              ) : (
                <p className="text-lg font-semibold uppercase text-text-1">{value}</p>
              )}
            </div>
          );
        })}
      </div>

      {/* API Info — values derived from env / status response where possible */}
      <div className="card">
        <div className="flex items-center gap-2 mb-4">
          <Info size={16} className="text-text-3" />
          <span className="label-tag">API Configuration</span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6">
          {[
            { label: 'API Base URL', value: import.meta.env.VITE_API_URL || '/api/v1', mono: true },
            { label: 'Version', value: import.meta.env.VITE_APP_VERSION || '2.0.0' },
            { label: 'Total Models', value: String(systemStatus?.total || 0) },
          ].map(({ label, value, mono }) => (
            <div key={label} className="flex justify-between py-2.5 border-b border-border-dim">
              <span className="text-sm text-text-2">{label}</span>
              <span className={`text-sm font-medium ${mono ? 'font-mono text-accent' : 'text-text-1'}`}>
                {value}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
