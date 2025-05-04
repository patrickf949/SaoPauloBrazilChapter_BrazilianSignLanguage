
import Home from './page.client'

export const metadata = {
  title: "Sign Language Recognition - Brazil Sao Paulo",
  description:
    "This is an website/app recognizes Brazilian sign language and translates it accordingly.",
  keywords: [
    "sign language",
    "translate",
    "brazil",
    "brazilian portuguese",
    "Reconhecimento",
    "de Língua de Sinais",
  ],
  openGraph: {
    title: "My Landing Page",
    description:
      "This is an website/app recognizes Brazilian sign language and translates it accordingly. Reconhecimento de Língua de Sinais",
    url: "https://example.com",
    siteName: "Sign Language Recognition - Brazil Sao Paulo",
    images: [
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "Sign Language Recognition - Brazil Sao Paulo",
    description:
      "This is an website/app recognizes Brazilian sign language and translates it accordingly. Reconhecimento de Língua de Sinais",
  },
};

export default function LandingPage() {
  return <Home />;
}
