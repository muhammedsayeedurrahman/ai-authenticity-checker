import axios from 'axios';

// Get base URL from env
// If VITE_API_URL is full URL with /api, we remove /api so we can safely use /api/analyze/... everywhere
let rawUrl = import.meta.env.VITE_API_URL || '';
if (rawUrl.endsWith('/api')) {
  rawUrl = rawUrl.slice(0, -4);
}

const api = axios.create({
  baseURL: rawUrl || '', // Fallback to empty string for Vite Proxy to work
});

export const forensicApi = {
  /**
   * Get the status of the loaded models from the backend
   * @returns {Promise<Object>} Status data
   */
  getStatus: async () => {
    try {
      const response = await api.get('/api/status');
      return response.data;
    } catch (error) {
      console.error('API Error (getStatus):', error);
      throw error;
    }
  },

  /**
   * Analyze an image
   * @param {File} file - The image file
   * @param {string} mode - 'Full Ensemble (7 models)' or 'Fast Mode (CorefakeNet)'
   * @returns {Promise<Object>}
   */
  analyzeImage: async (file, mode = 'Full Ensemble (7 models)') => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('mode', mode);
    
    // We point to /v1/analyze/image if using the main.py standard or /analyze/image based on app.py
    // The user instruction mentioned: Backend file reference: app.py (already implemented API routes like /api/analyze/image)
    // So we use /analyze/image
    try {
      const response = await api.post('/api/analyze/image', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('API Error (analyzeImage):', error);
      throw error;
    }
  },

  /**
   * Analyze a video
   * @param {File} file - The video file
   * @param {number} fps - Sampling FPS
   * @param {string} aggregation - Aggregation mode
   * @returns {Promise<Object>}
   */
  analyzeVideo: async (file, fps = 6, aggregation = 'weighted_avg') => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('fps', fps);
    formData.append('aggregation', aggregation);
    
    try {
      const response = await api.post('/api/analyze/video', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('API Error (analyzeVideo):', error);
      throw error;
    }
  },

  /**
   * Analyze audio
   * @param {File} file - The audio file
   * @returns {Promise<Object>}
   */
  analyzeAudio: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    
    try {
      const response = await api.post('/api/analyze/audio', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('API Error (analyzeAudio):', error);
      throw error;
    }
  },

  /**
   * Perform multimodal analysis
   * @param {File} image - Optional image file
   * @param {File} video - Optional video file
   * @param {File} audio - Optional audio file
   * @returns {Promise<Object>}
   */
  analyzeMultimodal: async (image, video, audio) => {
    const formData = new FormData();
    if (image) formData.append('image', image);
    if (video) formData.append('video', video);
    if (audio) formData.append('audio', audio);
    
    try {
      const response = await api.post('/api/analyze/multimodal', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('API Error (analyzeMultimodal):', error);
      throw error;
    }
  }
};
