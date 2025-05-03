import { create } from 'zustand';

export const useTranslationStore = create((set) => ({
  Information: {
    label: '',
    video: '',
  },
  loading: false,
  setLoading: (payload) => set(() => ({ loading: payload })),
  updateTranslation: (payload) => set(() => ({ user: payload })),
  resetTranslation:()=> set(() => ({ Information: { label: '', video: ''}})),
}));
