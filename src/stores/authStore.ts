import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import * as api from '../lib/api';

type AuthStatus = 'idle' | 'loading' | 'authenticated' | 'error';

type AuthState = {
  accessToken: string | null;
  refreshToken: string | null;
  userId: string | number | null;
  status: AuthStatus;
  error: string | null;
  isAuthenticated: boolean;
  register: (payload: api.AuthRegisterRequest) => Promise<string | number | undefined>;
  login: (payload: api.AuthLoginRequest) => Promise<void>;
  logout: () => Promise<void>;
  refreshAccessToken: () => Promise<void>;
  clearError: () => void;
};

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      accessToken: null,
      refreshToken: null,
      userId: null,
      status: 'idle',
      error: null,
      isAuthenticated: false,
      register: async (payload) => {
        set({ status: 'loading', error: null });
        try {
          const result = await api.register(payload);
          set({ status: 'idle', userId: result.userId ?? null });
          return result.userId;
        } catch (error) {
          set({ status: 'error', error: toErrorMessage(error) });
          throw error;
        }
      },
      login: async (payload) => {
        set({ status: 'loading', error: null });
        try {
          const session = await api.login(payload);
          localStorage.setItem('accessToken', session.accessToken);
          set({
            accessToken: session.accessToken,
            refreshToken: session.refreshToken ?? null,
            userId: session.userId ?? null,
            status: 'authenticated',
            error: null,
            isAuthenticated: true,
          });
        } catch (error) {
          set({ status: 'error', error: toErrorMessage(error), isAuthenticated: false });
          throw error;
        }
      },
      logout: async () => {
        const token = get().accessToken;
        set({ status: 'loading', error: null });
        try {
          if (token) await api.logout(token);
        } finally {
          localStorage.removeItem('accessToken');
          set({
            accessToken: null,
            refreshToken: null,
            userId: null,
            status: 'idle',
            error: null,
            isAuthenticated: false,
          });
        }
      },
      refreshAccessToken: async () => {
        const refreshToken = get().refreshToken;
        if (!refreshToken) throw new Error('저장된 refresh token이 없습니다.');

        set({ status: 'loading', error: null });
        try {
          const accessToken = await api.refreshAuthToken(refreshToken);
          set({ accessToken, status: 'authenticated', isAuthenticated: true });
        } catch (error) {
          set({ status: 'error', error: toErrorMessage(error), isAuthenticated: false });
          throw error;
        }
      },
      clearError: () => set({ error: null }),
    }),
    {
      name: 'solomon-auth',
      partialize: ({ accessToken, refreshToken, userId, isAuthenticated }) => ({
        accessToken,
        refreshToken,
        userId,
        isAuthenticated,
      }),
      onRehydrateStorage: () => (state) => {
        if (state?.accessToken) {
          state.status = 'authenticated';
          state.isAuthenticated = true;
        }
      },
    },
  ),
);

function toErrorMessage(error: unknown) {
  return error instanceof Error ? error.message : '요청 처리 중 오류가 발생했습니다.';
}
