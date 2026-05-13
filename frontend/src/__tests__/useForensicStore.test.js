import { describe, it, expect, vi, beforeEach } from 'vitest';
import { act } from '@testing-library/react';

// Mock supabase
vi.mock('../services/supabase', () => ({
  supabase: null,
  isAuthEnabled: () => false,
}));

// Mock the API
vi.mock('../services/api', () => ({
  forensicApi: {
    getStatus: vi.fn(),
    getHealth: vi.fn(),
    getHistory: vi.fn(),
    analyzeImage: vi.fn(),
    analyzeVideo: vi.fn(),
    analyzeAudio: vi.fn(),
    analyzeMultimodal: vi.fn(),
  },
}));

describe('useForensicStore', () => {
  let useForensicStore;
  let forensicApi;

  beforeEach(async () => {
    vi.clearAllMocks();
    // Re-import to get fresh store
    const storeModule = await import('../store/useForensicStore');
    useForensicStore = storeModule.default;
    const apiModule = await import('../services/api');
    forensicApi = apiModule.forensicApi;
  });

  it('has correct initial state', () => {
    const state = useForensicStore.getState();
    expect(state.systemStatus.loaded_models).toEqual([]);
    expect(state.history).toEqual([]);
    expect(state.imageAnalysis.isAnalyzing).toBe(false);
    expect(state.imageAnalysis.results).toBeNull();
    expect(state.imageAnalysis.error).toBeNull();
  });

  it('fetchStatus updates system status on success', async () => {
    forensicApi.getStatus.mockResolvedValue({
      loaded: ['vit', 'efficientnet'],
      missing: ['dino'],
      total: 2,
      corefakenet_ready: false,
    });

    await act(async () => {
      await useForensicStore.getState().fetchStatus();
    });

    const state = useForensicStore.getState();
    expect(state.systemStatus.loaded_models).toEqual(['vit', 'efficientnet']);
    expect(state.isStatusLoading).toBe(false);
    expect(state.statusError).toBeNull();
  });

  it('fetchHistory updates history on success', async () => {
    forensicApi.getHistory.mockResolvedValue({
      data: [{ id: '1', verdict: 'LIKELY_AUTHENTIC' }],
      total: 1,
    });

    await act(async () => {
      await useForensicStore.getState().fetchHistory();
    });

    const state = useForensicStore.getState();
    expect(state.history).toHaveLength(1);
    expect(state.historyTotal).toBe(1);
  });

  it('clearAnalysis resets analysis state', () => {
    act(() => {
      useForensicStore.getState().clearAnalysis('image');
    });

    const state = useForensicStore.getState();
    expect(state.imageAnalysis.isAnalyzing).toBe(false);
    expect(state.imageAnalysis.results).toBeNull();
    expect(state.imageAnalysis.error).toBeNull();
  });

  it('runImageAnalysis sets error on failure', async () => {
    forensicApi.analyzeImage.mockRejectedValue(new Error('Network error'));

    await act(async () => {
      await useForensicStore.getState().runImageAnalysis(new File([], 'test.jpg'));
    });

    const state = useForensicStore.getState();
    expect(state.imageAnalysis.isAnalyzing).toBe(false);
    expect(state.imageAnalysis.error).toBe('Network error');
  });
});
