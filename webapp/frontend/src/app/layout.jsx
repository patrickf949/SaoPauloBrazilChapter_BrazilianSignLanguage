"use client";
import { Roboto } from "next/font/google";
import ThemeRegistry from "@/components/ThemeRegistry";
import GlobalToast from "@/components/GlobalToast";
import "./globals.css";

const roboto = Roboto({
  subsets: ["latin"],
  weight: ["300", "400", "500", "700"],
});

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <meta name="viewport" content="initial-scale=1, width=device-width" />
        <link rel="icon" type="image/ico" href="favicon.ico" />
      </head>
      <body className={`${roboto.className} antialiased`}>
        <ThemeRegistry>
          {children}
          <GlobalToast />
        </ThemeRegistry>
      </body>
    </html>
  );
}
