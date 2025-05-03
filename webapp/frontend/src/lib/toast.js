import { useToastStore } from '../store/toastStore';

export const toast = {
  success: (msg) => useToastStore.getState().show({ message: msg, severity: 'success' }),
  error: (msg) => useToastStore.getState().show({ message: msg, severity: 'error' }),
  info: (msg) => useToastStore.getState().show({ message: msg, severity: 'info' }),
  warning: (msg) => useToastStore.getState().show({ message: msg, severity: 'warning' }),
};
