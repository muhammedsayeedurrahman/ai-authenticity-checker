import axios from 'axios';
import { supabase, isAuthEnabled } from './supabase';

let rawUrl = import.meta.env.VITE_API_URL || '';
if (rawUrl.endsWith('/api')) {
  rawUrl = rawUrl.slice(0, -4);
}

const api = axios.create({
  baseURL: rawUrl || '',
});

// Attach Supabase JWT token to all requests when auth is enabled
api.interceptors.request.use(async (config) => {
  if (!isAuthEnabled()) return config;

  const { data: { session } } = await supabase.auth.getSession();
  if (session?.access_token) {
    config.headers.Authorization = `Bearer ${session.access_token}`;
  }
  return config;
});

// Auto-refresh on 401
api.interceptors.response.use(
  (response) => response,
  async (error) => {
    const originalRequest = error.config;
    if (error.response?.status === 401 && !originalRequest._retry && isAuthEnabled()) {
      originalRequest._retry = true;
      const { data: { session } } = await supabase.auth.refreshSession();
      if (session?.access_token) {
        originalRequest.headers.Authorization = `Bearer ${session.access_token}`;
        return api(originalRequest);
      }
    }
    return Promise.reject(error);
  },
);

export const forensicApi = {
  getStatus: async () => {
    const response = await api.get('/api/v1/models/status');
    return response.data;
  },

  getHealth: async () => {
    const response = await api.get('/api/v1/health');
    return response.data;
  },

  analyzeImage: async (file, mode = 'ensemble', { signal } = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    const modeParam = mode.toLowerCase().includes('fast') ? 'fast' : 'ensemble';
    const response = await api.post(`/api/v1/analyze/image?mode=${modeParam}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      signal,
    });
    return response.data;
  },

  analyzeVideo: async (file, fps = 1, aggregation = 'weighted_avg', mode = 'ensemble', { signal } = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post(
      `/api/v1/analyze/video?fps=${fps}&aggregation=${aggregation}&mode=${mode}`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' }, signal },
    );
    return response.data;
  },

  analyzeAudio: async (file, { signal } = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/api/v1/analyze/audio', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      signal,
    });
    return response.data;
  },

  analyzeMultimodal: async (image, video, audio, { signal } = {}) => {
    const formData = new FormData();
    if (image) formData.append('image', image);
    if (video) formData.append('video', video);
    if (audio) formData.append('audio', audio);
    const response = await api.post('/api/v1/analyze/multimodal', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      signal,
    });
    return response.data;
  },

  getHistory: async (limit = 20, mediaType = null) => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (mediaType) params.set('media_type', mediaType);
    const response = await api.get(`/api/v1/history?${params}`);
    return response.data;
  },

  getAnalysis: async (id) => {
    const response = await api.get(`/api/v1/history/${id}`);
    return response.data;
  },
};

// Compliance & traceability API (India IT Rules 2026 feature) — orgs,
// org-scoped API keys, content labeling/SLA, audit trail, webhooks.
// Requires Supabase auth (isAuthEnabled()) — org creation/listing is
// JWT-only, matching the backend (see api/compliance_routes.py).
export const complianceApi = {
  listMyOrgs: async () => {
    const response = await api.get('/api/v1/compliance/orgs/me');
    return response.data;
  },

  createOrg: async (name, slug, contactEmail = '') => {
    const response = await api.post('/api/v1/compliance/orgs', {
      name, slug, contact_email: contactEmail,
    });
    return response.data;
  },

  listApiKeys: async (orgId) => {
    const response = await api.get(`/api/v1/compliance/orgs/${orgId}/api-keys`);
    return response.data;
  },

  createApiKey: async (orgId, label = '') => {
    const response = await api.post(`/api/v1/compliance/orgs/${orgId}/api-keys`, { label });
    return response.data;
  },

  revokeApiKey: async (orgId, keyId) => {
    const response = await api.post(`/api/v1/compliance/orgs/${orgId}/api-keys/${keyId}/revoke`);
    return response.data;
  },

  listSlaClocks: async (orgId, status = null) => {
    const params = status ? `?status=${status}` : '';
    const response = await api.get(`/api/v1/compliance/sla${params}`, {
      headers: { 'X-Proofyx-Org-Id': orgId },
    });
    return response.data;
  },

  recordContentAction: async (orgId, labelId, action, notes = '') => {
    const response = await api.post(
      `/api/v1/compliance/content/${labelId}/action`,
      { action, notes },
      { headers: { 'X-Proofyx-Org-Id': orgId } },
    );
    return response.data;
  },

  getAuditLog: async (orgId, limit = 100) => {
    const response = await api.get(`/api/v1/compliance/audit-log?limit=${limit}`, {
      headers: { 'X-Proofyx-Org-Id': orgId },
    });
    return response.data;
  },

  listWebhooks: async (orgId) => {
    const response = await api.get(`/api/v1/compliance/orgs/${orgId}/webhooks`);
    return response.data;
  },

  createWebhook: async (orgId, url, eventTypes = []) => {
    const response = await api.post(`/api/v1/compliance/orgs/${orgId}/webhooks`, {
      url, event_types: eventTypes,
    });
    return response.data;
  },

  revokeWebhook: async (orgId, endpointId) => {
    const response = await api.post(
      `/api/v1/compliance/orgs/${orgId}/webhooks/${endpointId}/revoke`,
    );
    return response.data;
  },

  testWebhook: async (orgId, endpointId) => {
    const response = await api.post(
      `/api/v1/compliance/orgs/${orgId}/webhooks/${endpointId}/test`,
    );
    return response.data;
  },
};
