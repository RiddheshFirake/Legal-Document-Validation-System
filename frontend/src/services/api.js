import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000,
});

// Add request interceptor for logging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Add response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

export const api = {
  // Health and status endpoints
  healthCheck: () => apiClient.get('/health'),
  getPipelineStatus: () => apiClient.get('/pipeline/status'),

  // Document validation
  validateDocument: (file, options = {}) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('options', JSON.stringify(options));
    formData.append('return_detailed', options.returnDetailed || false);
    formData.append('async_processing', options.asyncProcessing || false);
    
    return apiClient.post('/validate', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  // Batch validation
  validateBatch: (files, options = {}) => {
    const formData = new FormData();
    files.forEach(file => {
      formData.append('files', file);
    });
    formData.append('options', JSON.stringify(options));
    formData.append('max_concurrent', options.maxConcurrent || 3);
    formData.append('return_detailed', options.returnDetailed || false);
    
    return apiClient.post('/validate/batch', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  // Job management
  getJobStatus: (jobId) => apiClient.get(`/status/${jobId}`),
  deleteJob: (jobId) => apiClient.delete(`/jobs/${jobId}`),
  listJobs: () => apiClient.get('/jobs'),

  // URL validation
  validateUrl: (url, options = {}) => {
    return apiClient.post('/validate/url', {
      url,
      options,
      return_detailed: options.returnDetailed || false,
    });
  },
};

export default api;
