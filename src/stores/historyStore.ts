import { create } from 'zustand';
import * as api from '../lib/api';
import { useAuthStore } from './authStore';

type HistoryStoreState = {
  records: api.HistoryRecord[];
  selectedRecord: api.HistoryRecord | null;
  isLoading: boolean;
  error: string | null;
  fetchHistory: () => Promise<void>;
  fetchHistoryDetail: (id: string) => Promise<api.HistoryRecord>;
  deleteHistory: (id: string) => Promise<void>;
  clearError: () => void;
};

export const useHistoryStore = create<HistoryStoreState>((set, get) => ({
  records: [],
  selectedRecord: null,
  isLoading: false,
  error: null,
  fetchHistory: async () => {
    const token = requireToken();
    set({ isLoading: true, error: null });
    try {
      const records = await api.getHistory(token);
      set({ records, isLoading: false });
    } catch (error) {
      set({ isLoading: false, error: toErrorMessage(error) });
      throw error;
    }
  },
  fetchHistoryDetail: async (id) => {
    const token = requireToken();
    set({ isLoading: true, error: null });
    try {
      const selectedRecord = await api.getHistoryDetail(id, token);
      set({ selectedRecord, isLoading: false });
      return selectedRecord;
    } catch (error) {
      set({ isLoading: false, error: toErrorMessage(error) });
      throw error;
    }
  },
  deleteHistory: async (id) => {
    const token = requireToken();
    set({ isLoading: true, error: null });
    try {
      await api.deleteHistoryRecord(id, token);
      set({
        records: get().records.filter((record) => record.id !== id),
        selectedRecord: get().selectedRecord?.id === id ? null : get().selectedRecord,
        isLoading: false,
      });
    } catch (error) {
      set({ isLoading: false, error: toErrorMessage(error) });
      throw error;
    }
  },
  clearError: () => set({ error: null }),
}));

function requireToken() {
  const token = useAuthStore.getState().accessToken;
  if (!token) throw new Error('로그인이 필요한 기능입니다.');
  return token;
}

function toErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : '기록 요청 중 오류가 발생했습니다.';
}
