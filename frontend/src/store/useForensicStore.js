import { create } from 'zustand';
import { forensicApi } from '../services/api';

const useForensicStore = create((set) => ({
  // System Status State
  systemStatus: {
    loaded_models: [],
    missing_models: [],
    corefakenet_available: false,
    fusion_mlp_available: false,
    vit_available: false,
    device: 'cpu'
  },
  isStatusLoading: true,
  statusError: null,

  // Global functions
  fetchStatus: async () => {
    set({ isStatusLoading: true, statusError: null });
    try {
      const data = await forensicApi.getStatus();
      set({ systemStatus: data, isStatusLoading: false });
    } catch (error) {
      set({ 
        statusError: error.response?.data?.error || error.message, 
        isStatusLoading: false 
      });
    }
  },
}));

export default useForensicStore;
