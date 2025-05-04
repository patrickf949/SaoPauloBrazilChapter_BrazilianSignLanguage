// components/ThemeRegistry.tsx
'use client';

import { CacheProvider } from '@emotion/react';
import { ThemeProvider, CssBaseline } from '@mui/material';
import createEmotionCache from '../lib/emotionCache';
import { createTheme } from '@mui/material/styles';
import { useState } from 'react';

const theme = createTheme(); // You can customize this


export default function ThemeRegistry({ children }) {
  const [emotionCache] = useState(()=>createEmotionCache());
  return (
    <CacheProvider value={emotionCache}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        {children}
      </ThemeProvider>
    </CacheProvider>
  );
}
