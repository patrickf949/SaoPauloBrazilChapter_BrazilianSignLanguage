{/* eslint-disable react/prop-types 
Path: webapp/frontend/src/components/GlobalToast.jsx
 This component is responsible for displaying global toast notifications.
 It uses Material-UI's Snackbar and Alert components to show messages.
 The toast notifications are managed using Zustand for state management.
 The component listens to the toast store for changes in the notification state
*/}
import React from 'react';
import { Snackbar, Alert } from '@mui/material';
import { useToastStore } from '../store/toastStore';

const GlobalToast = () => {
  const { open, message, severity, hide } = useToastStore();

  return (
    <Snackbar
      open={open}
      autoHideDuration={5000}
      onClose={hide}
      anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
    >
      <Alert onClose={hide} severity={severity} variant="filled" sx={{ width: '100%' }}>
        {message}
      </Alert>
    </Snackbar>
  );
};

export default GlobalToast;