import { create } from 'zustand';

export const useTranslationStore = create((set) => ({
  info: {
    videoUrl: '',
    video: null,
    label: null,
  },
  loading: false,
  loadingVideo: false,
  setLoading: (payload) => set(() => ({ loading: payload })),
  setVideo: (payload) => set(() => ({ info: payload })),
  resetVideo:()=> set(() => ({ info: { label: '', video: null}})),
  setLoadingVideo: (payload) => set(() => ({ loadingVideo: payload })), 
}));
