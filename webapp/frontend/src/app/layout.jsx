/* 
This file is responsible for the root layout of the application.
It uses Material-UI for styling and theming.
*/
'use client';
import * as React from 'react';
import { AppRouterCacheProvider } from '@mui/material-nextjs/v15-appRouter';
import { ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import GlobalToast from '@/components/GlobalToast';
import theme from '@/theme';

export default function RootLayout(props) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <AppRouterCacheProvider options={{ enableCssLayer: true }}>
          <ThemeProvider theme={theme}>
            {/* CssBaseline kickstart an elegant, consistent, and simple baseline to build upon. */}
            <CssBaseline />
            {props.children}
            <GlobalToast />
          </ThemeProvider>
        </AppRouterCacheProvider>
      </body>
    </html>
  );
}