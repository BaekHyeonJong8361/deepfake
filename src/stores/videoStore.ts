import { create } from 'zustand';
import * as api from '../lib/api';
import { useAuthStore } from './authStore';

type VideoStoreState = {
  currentVideoId: string | null;
  targetLabel: string | null;
  status: api.VideoStatus | null;
  result: api.AnalysisResult | null;
  isSubmitting: boolean;
  isPolling: boolean;
  error: string | null;
  submitVideoUrl: (url: string, type: api.VideoAnalysisType) => Promise<string>;
  uploadVideoFile: (file: File, type: api.VideoAnalysisType) => Promise<string>;
  fetchStatus: (videoId?: string) => Promise<api.VideoStatus>;
  fetchResult: (videoId?: string) => Promise<api.AnalysisResult>;
  resetAnalysis: () => void;
  clearError: () => void;
};

export const useVideoStore = create<VideoStoreState>((set, get) => ({
  currentVideoId: null,
  targetLabel: null,
  status: null,
  result: null,
  isSubmitting: false,
  isPolling: false,
  error: null,
  submitVideoUrl: async (url, type) => {
    set({ isSubmitting: true, error: null, result: null, status: 'queued', targetLabel: url });
    try {
      const videoId = await api.requestVideoUrl(url, type, getAccessToken());
      set({ currentVideoId: videoId, status: 'queued', isSubmitting: false });
      return videoId;
    } catch (error) {
      set({ isSubmitting: false, status: 'failed', error: toErrorMessage(error) });
      throw error;
    }
  },
  uploadVideoFile: async (file, type) => {
    set({ isSubmitting: true, error: null, result: null, status: 'queued', targetLabel: file.name });
    try {
      const videoId = await api.uploadVideo(file, type, getAccessToken());
      set({ currentVideoId: videoId, status: 'queued', isSubmitting: false });
      return videoId;
    } catch (error) {
      set({ isSubmitting: false, status: 'failed', error: toErrorMessage(error) });
      throw error;
    }
  },
  fetchStatus: async (videoId = get().currentVideoId ?? '') => {
    if (!videoId) throw new Error('Analysis videoId is missing.');

    set({ isPolling: true, error: null });
    try {
      const status = await api.getVideoStatus(videoId, getAccessToken());
      set({ status, isPolling: false });
      return status;
    } catch (error) {
      set({ status: 'failed', isPolling: false, error: toErrorMessage(error) });
      throw error;
    }
  },
  fetchResult: async (videoId = get().currentVideoId ?? '') => {
    if (!videoId) throw new Error('Result videoId is missing.');

    set({ isPolling: true, error: null });
    try {
      const result = await api.getVideoResult(videoId, getAccessToken());
      set({ result, status: 'completed', isPolling: false });
      return result;
    } catch (error) {
      set({ status: 'failed', isPolling: false, error: toErrorMessage(error) });
      throw error;
    }
  },
  resetAnalysis: () => set({ currentVideoId: null, targetLabel: null, status: null, result: null, error: null }),
  clearError: () => set({ error: null }),
}));

function getAccessToken() {
  return useAuthStore.getState().accessToken;
}

function toErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : 'Video analysis request failed.';
}
