import { create } from 'zustand';
import { forensicApi } from '../services/api';

const createAnalysisSlice = () => ({
  isAnalyzing: false,
  results: null,
  error: null,
});

const useForensicStore = create((set, get) => ({
  // System Status State
  systemStatus: {
    loaded_models: [],
    missing_models: [],
    corefakenet_available: false,
    fusion_mlp_available: false,
    vit_available: false,
    device: 'cpu',
  },
  isStatusLoading: true,
  statusError: null,

  // History State
  history: [],
  historyTotal: 0,
  isHistoryLoading: false,
  historyError: null,

  // Per-page analysis state (persists across navigation)
  imageAnalysis: createAnalysisSlice(),
  videoAnalysis: createAnalysisSlice(),
  audioAnalysis: createAnalysisSlice(),
  multimodalAnalysis: createAnalysisSlice(),

  // Analysis actions
  runImageAnalysis: async (file, mode) => {
    set({ imageAnalysis: { isAnalyzing: true, results: null, error: null } });
    try {
      const data = await forensicApi.analyzeImage(file, mode);
      if (data.success) {
        set({ imageAnalysis: { isAnalyzing: false, results: data, error: null } });
      } else {
        set({ imageAnalysis: { isAnalyzing: false, results: null, error: data.error || 'Analysis failed' } });
      }
    } catch (err) {
      set({ imageAnalysis: { isAnalyzing: false, results: null, error: err.response?.data?.error || err.message || 'An error occurred' } });
    }
  },

  runVideoAnalysis: async (file, fps, aggregation) => {
    set({ videoAnalysis: { isAnalyzing: true, results: null, error: null } });
    try {
      const data = await forensicApi.analyzeVideo(file, fps, aggregation);
      if (data.success) {
        set({ videoAnalysis: { isAnalyzing: false, results: data, error: null } });
      } else {
        set({ videoAnalysis: { isAnalyzing: false, results: null, error: data.error || 'Analysis failed' } });
      }
    } catch (err) {
      set({ videoAnalysis: { isAnalyzing: false, results: null, error: err.response?.data?.error || err.message || 'An error occurred' } });
    }
  },

  runAudioAnalysis: async (file) => {
    set({ audioAnalysis: { isAnalyzing: true, results: null, error: null } });
    try {
      const data = await forensicApi.analyzeAudio(file);
      if (data.success) {
        set({ audioAnalysis: { isAnalyzing: false, results: data, error: null } });
      } else {
        set({ audioAnalysis: { isAnalyzing: false, results: null, error: data.error || 'Analysis failed' } });
      }
    } catch (err) {
      set({ audioAnalysis: { isAnalyzing: false, results: null, error: err.response?.data?.error || err.message || 'An error occurred' } });
    }
  },

  runMultimodalAnalysis: async (image, video, audio) => {
    set({ multimodalAnalysis: { isAnalyzing: true, results: null, error: null } });
    try {
      const data = await forensicApi.analyzeMultimodal(image, video, audio);
      if (data.success) {
        set({ multimodalAnalysis: { isAnalyzing: false, results: data, error: null } });
      } else {
        set({ multimodalAnalysis: { isAnalyzing: false, results: null, error: data.error || 'Analysis failed' } });
      }
    } catch (err) {
      set({ multimodalAnalysis: { isAnalyzing: false, results: null, error: err.response?.data?.error || err.message || 'An error occurred' } });
    }
  },

  clearAnalysis: (type) => {
    set({ [`${type}Analysis`]: createAnalysisSlice() });
  },

  fetchStatus: async () => {
    set({ isStatusLoading: true, statusError: null });
    try {
      const data = await forensicApi.getStatus();
      set({
        systemStatus: {
          loaded_models: data.loaded || [],
          missing_models: data.missing || [],
          corefakenet_available: data.corefakenet_ready || false,
          fusion_mlp_available: (data.loaded || []).some(
            (m) => m.toLowerCase().includes('fusion') || m.toLowerCase().includes('mlp'),
          ),
          vit_available: (data.loaded || []).some(
            (m) => m.toLowerCase().includes('vit'),
          ),
          device: 'auto',
          total: data.total || 0,
        },
        isStatusLoading: false,
      });
    } catch (error) {
      set({
        statusError: error.response?.data?.error || error.message,
        isStatusLoading: false,
      });
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
      set({
        historyError: error.response?.data?.error || error.message,
        isHistoryLoading: false,
      });
    }
  },
}));

export default useForensicStore;
