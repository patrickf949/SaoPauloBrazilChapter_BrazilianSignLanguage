import { create } from "zustand";

export const useTranslationStore = create((set) => ({
  info: {
    videoUrl: "",
    video: null,
    label: null,
  },
  loading: false,
  loadingVideo: false,
  result: {
    video: null,
    videoUrl: null,
    label: null,
  },

  // State setters
  setLoading: (payload) => set(() => ({ loading: payload })),
  setVideo: (payload) => set(() => ({ info: payload })),
  setResult: (payload) => set(() => ({ result: payload })),
  resetVideo: () => set(() => ({ info: { label: "", video: null, videoUrl: "" } })),
  resetResult: () => set(() => ({ result: { label: null, video: null, videoUrl: null } })),
  setLoadingVideo: (payload) => set(() => ({ loadingVideo: payload })),
}));
