import { create } from 'zustand';

export const useTranslationStore = create((set) => ({
  info: {
    videoUrl: '',
    video: null,
    label: null,
  },
  loading: false,
  loadingVideo: false,
  result: {
    video:null,
    videoUrl:null,
    label:null,
  },
  setLoading: (payload) => set(() => ({ loading: payload })),
  setVideo: (payload) => set(() => ({ info: payload })),
  setResult: (payload) => set(()=> ({result})),
  resetVideo:()=> set(() => ({ info: { label: '', video: null}})),
  setLoadingVideo: (payload) => set(() => ({ loadingVideo: payload })), 
}));
