import { create } from 'zustand';
import axios from 'axios';
import { forensicApi } from '../services/api';
import useToastStore from './useToastStore';

const createAnalysisSlice = () => ({
  isAnalyzing: false,
  results: null,
  error: null,
});

// AbortControllers live outside Zustand state (non-serializable).
const abortControllers = new Map();

/**
 * Normalize an API envelope response into a flat results object.
 *
 * The API returns `{ success, data: { risk_score, ... }, error }`.
 * Previously, stores saved the raw envelope and components had to guess
 * between `results.risk_percentage`, `results.data?.risk_percent`, etc.
 *
 * Now the store always unwraps `data` so components get a flat shape:
 *   results.risk_percent, results.verdict, results.model_scores, ...
 */
function normalizeResults(apiResponse) {
  if (!apiResponse) return null;

  // If the API returned {success, data: {...}}, unwrap `data`
  const raw = apiResponse.data || apiResponse;

  // Ensure risk_percent always exists (some endpoints return risk_score 0-1)
  const riskScore = raw.risk_score ?? 0;
  const riskPercent = raw.risk_percent ?? riskScore * 100;

  return {
    ...raw,
    risk_score: riskScore,
    risk_percent: riskPercent,
    // Legacy aliases that some components referenced
    risk_percentage: riskPercent,
    authenticity_percentage: raw.authenticity_score ?? (100 - riskPercent),
  };
}

function extractError(err) {
  return err.response?.data?.error || err.response?.data?.detail || err.message || 'An error occurred';
}

const useForensicStore = create((set) => ({
  // System Status
  systemStatus: {
    loaded_models: [],
    missing_models: [],
    corefakenet_available: false,
    fusion_mlp_available: false,
    vit_available: false,
    device: 'cpu',
    total: 0,
    reverse_search_available: false,
  },
  isStatusLoading: true,
  statusError: null,

  // History
  history: [],
  historyTotal: 0,
  isHistoryLoading: false,
  historyError: null,

  // Per-page analysis state (persists across navigation)
  imageAnalysis: createAnalysisSlice(),
  videoAnalysis: createAnalysisSlice(),
  audioAnalysis: createAnalysisSlice(),
  documentAnalysis: createAnalysisSlice(),
  multimodalAnalysis: createAnalysisSlice(),

  // Pending file from drag-drop on Dashboard
  pendingFile: null,
  setPendingFile: (file) => set({ pendingFile: file }),
  clearPendingFile: () => set({ pendingFile: null }),

  // --- Analysis Actions ---

  runImageAnalysis: async (file, mode, reverseSearch = false) => {
    abortControllers.get('image')?.abort();
    const controller = new AbortController();
    abortControllers.set('image', controller);
    set({ imageAnalysis: { isAnalyzing: true, results: null, error: null } });
    try {
      const data = await forensicApi.analyzeImage(file, mode, reverseSearch, { signal: controller.signal });
      if (data.success) {
        set({ imageAnalysis: { isAnalyzing: false, results: normalizeResults(data), error: null } });
        useToastStore.getState().addToast('Image analysis complete', 'success');
      } else {
        set({ imageAnalysis: { isAnalyzing: false, results: null, error: data.error || 'Analysis failed' } });
        useToastStore.getState().addToast(data.error || 'Image analysis failed', 'error');
      }
    } catch (err) {
      if (axios.isCancel(err) || err.name === 'AbortError' || err.name === 'CanceledError') return;
      set({ imageAnalysis: { isAnalyzing: false, results: null, error: extractError(err) } });
      useToastStore.getState().addToast(extractError(err), 'error');
    } finally {
      abortControllers.delete('image');
    }
  },

  runVideoAnalysis: async (file, fps, aggregation, mode = 'ensemble') => {
    abortControllers.get('video')?.abort();
    const controller = new AbortController();
    abortControllers.set('video', controller);
    set({ videoAnalysis: { isAnalyzing: true, results: null, error: null } });
    try {
      const data = await forensicApi.analyzeVideo(file, fps, aggregation, mode, { signal: controller.signal });
      if (data.success) {
        set({ videoAnalysis: { isAnalyzing: false, results: normalizeResults(data), error: null } });
        useToastStore.getState().addToast('Video analysis complete', 'success');
      } else {
        set({ videoAnalysis: { isAnalyzing: false, results: null, error: data.error || 'Analysis failed' } });
        useToastStore.getState().addToast(data.error || 'Video analysis failed', 'error');
      }
    } catch (err) {
      if (axios.isCancel(err) || err.name === 'AbortError' || err.name === 'CanceledError') return;
      set({ videoAnalysis: { isAnalyzing: false, results: null, error: extractError(err) } });
      useToastStore.getState().addToast(extractError(err), 'error');
    } finally {
      abortControllers.delete('video');
    }
  },

  runAudioAnalysis: async (file) => {
    abortControllers.get('audio')?.abort();
    const controller = new AbortController();
    abortControllers.set('audio', controller);
    set({ audioAnalysis: { isAnalyzing: true, results: null, error: null } });
    try {
      const data = await forensicApi.analyzeAudio(file, { signal: controller.signal });
      if (data.success) {
        set({ audioAnalysis: { isAnalyzing: false, results: normalizeResults(data), error: null } });
        useToastStore.getState().addToast('Audio analysis complete', 'success');
      } else {
        set({ audioAnalysis: { isAnalyzing: false, results: null, error: data.error || 'Analysis failed' } });
        useToastStore.getState().addToast(data.error || 'Audio analysis failed', 'error');
      }
    } catch (err) {
      if (axios.isCancel(err) || err.name === 'AbortError' || err.name === 'CanceledError') return;
      set({ audioAnalysis: { isAnalyzing: false, results: null, error: extractError(err) } });
      useToastStore.getState().addToast(extractError(err), 'error');
    } finally {
      abortControllers.delete('audio');
    }
  },

  runDocumentAnalysis: async (file, idType = '', idNumber = '', reverseSearch = false) => {
    abortControllers.get('document')?.abort();
    const controller = new AbortController();
    abortControllers.set('document', controller);
    set({ documentAnalysis: { isAnalyzing: true, results: null, error: null } });
    try {
      const data = await forensicApi.analyzeDocument(file, idType, idNumber, reverseSearch, { signal: controller.signal });
      if (data.success) {
        set({ documentAnalysis: { isAnalyzing: false, results: normalizeResults(data), error: null } });
        useToastStore.getState().addToast('Document analysis complete', 'success');
      } else {
        set({ documentAnalysis: { isAnalyzing: false, results: null, error: data.error || 'Analysis failed' } });
        useToastStore.getState().addToast(data.error || 'Document analysis failed', 'error');
      }
    } catch (err) {
      if (axios.isCancel(err) || err.name === 'AbortError' || err.name === 'CanceledError') return;
      set({ documentAnalysis: { isAnalyzing: false, results: null, error: extractError(err) } });
      useToastStore.getState().addToast(extractError(err), 'error');
    } finally {
      abortControllers.delete('document');
    }
  },

  runMultimodalAnalysis: async (image, video, audio) => {
    abortControllers.get('multimodal')?.abort();
    const controller = new AbortController();
    abortControllers.set('multimodal', controller);
    set({ multimodalAnalysis: { isAnalyzing: true, results: null, error: null } });
    try {
      const data = await forensicApi.analyzeMultimodal(image, video, audio, { signal: controller.signal });
      if (data.success) {
        set({ multimodalAnalysis: { isAnalyzing: false, results: normalizeResults(data), error: null } });
        useToastStore.getState().addToast('Multimodal analysis complete', 'success');
      } else {
        set({ multimodalAnalysis: { isAnalyzing: false, results: null, error: data.error || 'Analysis failed' } });
        useToastStore.getState().addToast(data.error || 'Multimodal analysis failed', 'error');
      }
    } catch (err) {
      if (axios.isCancel(err) || err.name === 'AbortError' || err.name === 'CanceledError') return;
      set({ multimodalAnalysis: { isAnalyzing: false, results: null, error: extractError(err) } });
      useToastStore.getState().addToast(extractError(err), 'error');
    } finally {
      abortControllers.delete('multimodal');
    }
  },

  cancelAnalysis: (type) => {
    const controller = abortControllers.get(type);
    if (controller) {
      controller.abort();
      abortControllers.delete(type);
    }
  },

  clearAnalysis: (type) => {
    set({ [`${type}Analysis`]: createAnalysisSlice() });
  },

  fetchStatus: async () => {
    set({ isStatusLoading: true, statusError: null });
    try {
      const data = await forensicApi.getStatus();
      const loaded = data.loaded || [];
      set({
        systemStatus: {
          loaded_models: loaded,
          missing_models: data.missing || [],
          corefakenet_available: data.corefakenet_ready || false,
          fusion_mlp_available: loaded.some(
            (m) => m.toLowerCase().includes('fusion') || m.toLowerCase().includes('mlp'),
          ),
          vit_available: loaded.some((m) => m.toLowerCase().includes('vit')),
          device: data.device || 'cpu',
          total: data.total || 0,
          reverse_search_available: data.reverse_search_available || false,
        },
        isStatusLoading: false,
      });
    } catch (error) {
      set({ statusError: extractError(error), isStatusLoading: false });
    }
  },

  fetchHistory: async (limit = 20, mediaType = null) => {
    set({ isHistoryLoading: true, historyError: null });
    try {
      const data = await forensicApi.getHistory(limit, mediaType);
      set({
        history: data.data || [],
        historyTotal: data.total || 0,
        isHistoryLoading: false,
      });
    } catch (error) {
      set({ historyError: extractError(error), isHistoryLoading: false });
    }
  },
}));

export default useForensicStore;
