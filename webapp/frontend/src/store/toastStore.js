import { create } from 'zustand';

export const useToastStore = create((set) => ({
  open: false,
  message: '',
  severity: 'success', // success | error | info | warning

  show: ({ message, severity = 'success' }) =>
    set({ open: true, message, severity }),
  hide: () => set({ open: false }),
}));
