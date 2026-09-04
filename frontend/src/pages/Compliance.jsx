import React, { useCallback, useEffect, useState } from 'react';
import { ShieldCheck, Plus, Key, Clock, ScrollText, Webhook, Copy, RefreshCw } from 'lucide-react';
import PageHeader from '../components/PageHeader';
import ConfirmDialog from '../components/ConfirmDialog';
import useAuthStore from '../store/useAuthStore';
import useToastStore from '../store/useToastStore';
import { isAuthEnabled } from '../services/supabase';
import { complianceApi } from '../services/api';

const SLA_STATUS_STYLES = {
  running: 'text-text-2',
  due_soon: 'text-risk-caution',
  breached: 'text-risk-critical',
  met: 'text-risk-clear',
  cancelled: 'text-text-3',
};

function SecretReveal({ label, value }) {
  const { addToast } = useToastStore();
  const copy = () => {
    navigator.clipboard?.writeText(value);
    addToast('Copied to clipboard — this is shown only once.', 'success');
  };
  return (
    <div className="inset-panel p-3 mt-2">
      <p className="label-tag mb-1">{label} (shown once — copy it now)</p>
      <div className="flex items-center gap-2">
        <code className="text-xs font-mono text-accent break-all">{value}</code>
        <button onClick={copy} className="btn-ghost px-2 py-1 flex-shrink-0" title="Copy">
          <Copy size={13} />
        </button>
      </div>
    </div>
  );
}

export default function Compliance() {
  const { user } = useAuthStore();
  const { addToast } = useToastStore();

  const [orgs, setOrgs] = useState([]);
  const [orgId, setOrgId] = useState('');
  const [loadingOrgs, setLoadingOrgs] = useState(true);
  const [newOrgName, setNewOrgName] = useState('');
  const [newOrgSlug, setNewOrgSlug] = useState('');

  const [apiKeys, setApiKeys] = useState([]);
  const [newKeySecret, setNewKeySecret] = useState(null);
  const [revokeTarget, setRevokeTarget] = useState(null);

  const [slaClocks, setSlaClocks] = useState([]);
  const [auditEntries, setAuditEntries] = useState([]);
  const [chainVerified, setChainVerified] = useState(true);
  const [webhooks, setWebhooks] = useState([]);
  const [newWebhookUrl, setNewWebhookUrl] = useState('');
  const [newWebhookSecret, setNewWebhookSecret] = useState(null);
  const [refreshing, setRefreshing] = useState(false);

  const loadOrgs = useCallback(async () => {
    setLoadingOrgs(true);
    try {
      const res = await complianceApi.listMyOrgs();
      setOrgs(res.data || []);
      if (!orgId && res.data?.length) setOrgId(res.data[0].id);
    } catch {
      addToast('Failed to load organizations.', 'error');
    } finally {
      setLoadingOrgs(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadOrgData = useCallback(async (id) => {
    if (!id) return;
    setRefreshing(true);
    try {
      const [keysRes, slaRes, auditRes, webhooksRes] = await Promise.all([
        complianceApi.listApiKeys(id),
        complianceApi.listSlaClocks(id),
        complianceApi.getAuditLog(id, 50),
        complianceApi.listWebhooks(id),
      ]);
      setApiKeys(keysRes.data || []);
      setSlaClocks(slaRes.data || []);
      setAuditEntries(auditRes.data || []);
      setChainVerified(auditRes.chain_verified);
      setWebhooks(webhooksRes.data || []);
    } catch {
      addToast('Failed to load compliance data for this organization.', 'error');
    } finally {
      setRefreshing(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => { if (isAuthEnabled() && user) loadOrgs(); }, [loadOrgs, user]);
  useEffect(() => { if (orgId) loadOrgData(orgId); }, [orgId, loadOrgData]);

  const handleCreateOrg = async (e) => {
    e.preventDefault();
    if (!newOrgName.trim() || !newOrgSlug.trim()) return;
    try {
      const res = await complianceApi.createOrg(newOrgName.trim(), newOrgSlug.trim());
      addToast(`Organization "${res.data.name}" created.`, 'success');
      setNewOrgName('');
      setNewOrgSlug('');
      await loadOrgs();
      setOrgId(res.data.id);
    } catch (err) {
      addToast(err.response?.data?.detail || 'Failed to create organization.', 'error');
    }
  };

  const handleCreateKey = async () => {
    try {
      const res = await complianceApi.createApiKey(orgId, 'Console-issued key');
      setNewKeySecret(res.raw_key);
      await loadOrgData(orgId);
    } catch {
      addToast('Failed to create API key.', 'error');
    }
  };

  const handleRevokeKey = async () => {
    if (!revokeTarget) return;
    try {
      await complianceApi.revokeApiKey(orgId, revokeTarget.id);
      addToast('API key revoked.', 'success');
      await loadOrgData(orgId);
    } catch {
      addToast('Failed to revoke API key.', 'error');
    } finally {
      setRevokeTarget(null);
    }
  };

  const handleCreateWebhook = async (e) => {
    e.preventDefault();
    if (!newWebhookUrl.trim()) return;
    try {
      const res = await complianceApi.createWebhook(orgId, newWebhookUrl.trim());
      setNewWebhookSecret(res.secret);
      setNewWebhookUrl('');
      await loadOrgData(orgId);
    } catch (err) {
      addToast(err.response?.data?.detail || 'Failed to register webhook (check the URL is https and publicly reachable).', 'error');
    }
  };

  const handleTestWebhook = async (endpointId) => {
    try {
      const res = await complianceApi.testWebhook(orgId, endpointId);
      addToast(
        res.delivered ? `Test delivery succeeded (HTTP ${res.status_code}).` : `Test delivery failed: ${res.error}`,
        res.delivered ? 'success' : 'error',
      );
    } catch {
      addToast('Failed to send test delivery.', 'error');
    }
  };

  const handleCloseSla = async (labelId, action) => {
    try {
      await complianceApi.recordContentAction(orgId, labelId, action);
      addToast('SLA clock updated.', 'success');
      await loadOrgData(orgId);
    } catch {
      addToast('Failed to record action.', 'error');
    }
  };

  if (!isAuthEnabled()) {
    return (
      <div className="space-y-6">
        <PageHeader icon={ShieldCheck} title="Compliance Console" subtitle="India IT Rules 2026 traceability & takedown-SLA tooling" />
        <div className="card">
          <p className="text-sm text-text-2">
            The compliance console requires sign-in (Supabase auth) — it isn't configured for this deployment.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <PageHeader
        icon={ShieldCheck}
        title="Compliance Console"
        subtitle="Organizations, API keys, takedown-SLA queue, and the tamper-evident audit trail — India IT Rules 2026"
        actions={orgId && (
          <button onClick={() => loadOrgData(orgId)} disabled={refreshing} className="btn-ghost flex items-center gap-2">
            <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} />
            {refreshing ? 'Syncing...' : 'Refresh'}
          </button>
        )}
      />

      {/* Org selector / creation */}
      <div className="card">
        <p className="label-tag mb-3">Organization</p>
        {loadingOrgs ? (
          <p className="text-sm text-text-3">Loading organizations...</p>
        ) : orgs.length > 0 ? (
          <select value={orgId} onChange={(e) => setOrgId(e.target.value)} className="field-input text-sm max-w-sm">
            {orgs.map((o) => (
              <option key={o.id} value={o.id}>{o.name} ({o.role})</option>
            ))}
          </select>
        ) : (
          <p className="text-sm text-text-2 mb-3">No organization yet — create one to start issuing compliance-scoped API keys.</p>
        )}

        <form onSubmit={handleCreateOrg} className="flex flex-wrap gap-2 mt-3">
          <input
            value={newOrgName}
            onChange={(e) => setNewOrgName(e.target.value)}
            placeholder="Organization name"
            className="field-input text-sm flex-1 min-w-[180px]"
          />
          <input
            value={newOrgSlug}
            onChange={(e) => setNewOrgSlug(e.target.value.toLowerCase().replace(/[^a-z0-9-]/g, '-'))}
            placeholder="url-slug"
            className="field-input text-sm w-40"
          />
          <button type="submit" className="btn-primary px-3 flex items-center gap-1.5">
            <Plus size={14} /> New Org
          </button>
        </form>
      </div>

      {orgId && (
        <>
          {/* API Keys */}
          <div className="card">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <Key size={14} className="text-text-3" />
                <span className="label-tag">Org-Scoped API Keys</span>
              </div>
              <button onClick={handleCreateKey} className="btn-ghost text-xs px-2 py-1">+ New Key</button>
            </div>
            {newKeySecret && <SecretReveal label="New API key" value={newKeySecret} />}
            <div className="space-y-1.5 mt-2">
              {apiKeys.length === 0 && <p className="text-sm text-text-3 italic">No API keys issued yet.</p>}
              {apiKeys.map((k) => (
                <div key={k.id} className="flex items-center justify-between p-2.5 rounded-lg bg-white/[0.02] border border-border-dim">
                  <div>
                    <span className="text-sm font-mono text-text-1">{k.key_prefix}...</span>
                    {k.label && <span className="text-xs text-text-3 ml-2">{k.label}</span>}
                    {k.revoked_at && <span className="text-xs text-risk-critical ml-2">revoked</span>}
                  </div>
                  {!k.revoked_at && (
                    <button onClick={() => setRevokeTarget(k)} className="text-xs text-risk-critical hover:underline">
                      Revoke
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* SLA Queue */}
          <div className="card">
            <div className="flex items-center gap-2 mb-3">
              <Clock size={14} className="text-text-3" />
              <span className="label-tag">Takedown SLA Queue</span>
            </div>
            <div className="space-y-1.5">
              {slaClocks.length === 0 && <p className="text-sm text-text-3 italic">No SLA clocks — nothing has been flagged.</p>}
              {slaClocks.map((c) => (
                <div key={c.id} className="flex items-center justify-between p-2.5 rounded-lg bg-white/[0.02] border border-border-dim">
                  <div>
                    <span className={`text-sm font-semibold uppercase ${SLA_STATUS_STYLES[c.effective_status] || 'text-text-2'}`}>
                      {c.effective_status.replace('_', ' ')}
                    </span>
                    <span className="text-xs text-text-3 ml-2 font-mono">due {c.due_at}</span>
                  </div>
                  {c.status === 'running' && (
                    <div className="flex gap-2">
                      <button onClick={() => handleCloseSla(c.content_label_id, 'removed')} className="text-xs text-accent hover:underline">
                        Mark removed
                      </button>
                      <button onClick={() => handleCloseSla(c.content_label_id, 'cleared_false_positive')} className="text-xs text-text-3 hover:underline">
                        False positive
                      </button>
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>

          {/* Webhooks */}
          <div className="card">
            <div className="flex items-center gap-2 mb-3">
              <Webhook size={14} className="text-text-3" />
              <span className="label-tag">Webhooks</span>
            </div>
            {newWebhookSecret && <SecretReveal label="Webhook HMAC secret" value={newWebhookSecret} />}
            <form onSubmit={handleCreateWebhook} className="flex gap-2 mt-2 mb-3">
              <input
                value={newWebhookUrl}
                onChange={(e) => setNewWebhookUrl(e.target.value)}
                placeholder="https://your-platform.example.com/proofyx-webhook"
                className="field-input text-sm flex-1"
              />
              <button type="submit" className="btn-primary px-3 text-sm">Register</button>
            </form>
            <div className="space-y-1.5">
              {webhooks.length === 0 && <p className="text-sm text-text-3 italic">No webhook endpoints registered.</p>}
              {webhooks.map((w) => (
                <div key={w.id} className="flex items-center justify-between p-2.5 rounded-lg bg-white/[0.02] border border-border-dim">
                  <span className="text-sm font-mono text-text-1 truncate max-w-[50%]">{w.url}</span>
                  <div className="flex items-center gap-3">
                    {!w.is_active && <span className="text-xs text-risk-critical">inactive</span>}
                    <button onClick={() => handleTestWebhook(w.id)} className="text-xs text-accent hover:underline">Send test</button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Audit log */}
          <div className="card">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <ScrollText size={14} className="text-text-3" />
                <span className="label-tag">Audit Trail</span>
              </div>
              <span className={`text-xs font-semibold ${chainVerified ? 'text-risk-clear' : 'text-risk-critical'}`}>
                {chainVerified ? 'Chain verified' : 'CHAIN BROKEN — tampering detected'}
              </span>
            </div>
            <div className="space-y-1 max-h-96 overflow-y-auto">
              {auditEntries.length === 0 && <p className="text-sm text-text-3 italic">No audit entries yet.</p>}
              {[...auditEntries].reverse().map((e) => (
                <div key={e.id} className="flex justify-between py-1.5 border-b border-border-dim text-xs">
                  <span className="font-mono text-text-2">{e.event_type}</span>
                  <span className="text-text-3">{e.occurred_at}</span>
                </div>
              ))}
            </div>
          </div>
        </>
      )}

      <ConfirmDialog
        open={!!revokeTarget}
        title="Revoke API Key"
        message={`Revoke key ${revokeTarget?.key_prefix}...? Requests using it will start failing immediately. This cannot be undone.`}
        confirmLabel="Revoke Key"
        onConfirm={handleRevokeKey}
        onCancel={() => setRevokeTarget(null)}
      />
    </div>
  );
}
