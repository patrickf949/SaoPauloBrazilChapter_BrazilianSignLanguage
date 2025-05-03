'use client';
import { Geist, Geist_Mono } from "next/font/google";
import ThemeRegistry from "@/components/ThemeRegistry";
import "./globals.css";
import '@fontsource/roboto/300.css';
import '@fontsource/roboto/400.css';
import '@fontsource/roboto/500.css';
import '@fontsource/roboto/700.css';


const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});





export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
      <meta name="viewport" content="initial-scale=1, width=device-width" />
      <link rel="icon" type="image/ico" href="favicon.ico"/>
      </head>

      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
         <ThemeRegistry>
            {children}
            <GlobalToast />
            </ThemeRegistry>
      </body>
    </html>
  );
}
