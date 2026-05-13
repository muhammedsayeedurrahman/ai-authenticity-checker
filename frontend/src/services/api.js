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

  analyzeImage: async (file, mode = 'ensemble') => {
    const formData = new FormData();
    formData.append('file', file);
    const modeParam = mode.toLowerCase().includes('fast') ? 'fast' : 'ensemble';
    const response = await api.post(`/api/v1/analyze/image?mode=${modeParam}`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  analyzeVideo: async (file, fps = 4, aggregation = 'weighted_avg') => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post(
      `/api/v1/analyze/video?fps=${fps}&aggregation=${aggregation}`,
      formData,
      { headers: { 'Content-Type': 'multipart/form-data' } },
    );
    return response.data;
  },

  analyzeAudio: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/api/v1/analyze/audio', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  },

  analyzeMultimodal: async (image, video, audio) => {
    const formData = new FormData();
    if (image) formData.append('image', image);
    if (video) formData.append('video', video);
    if (audio) formData.append('audio', audio);
    const response = await api.post('/api/v1/analyze/multimodal', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
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
