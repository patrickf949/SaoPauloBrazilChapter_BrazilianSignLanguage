
import Home from './page.client'

export const metadata = {
  title: "AI for Brazilian Sign Language Translation - Demo Application",
  description:
    "This is the demo application for an AI model developed to translate " +
    "Brazilian Sign Language in a healthcare context. It was developed as " +
    "part of an Omdena local chapter challenge with the São Paulo chapter." +
    "For more details about the project and our achievements, please visit our Project Page website." +
    "The AI model was trained to recognize 25 different signs." +
    "Below you can test it out on some sample sign videos from our dataset, or record & upload your own!",
  keywords: [
    "sign language",
    "translate",
    "brazil",
    "brazilian portuguese",
    "Reconhecimento",
    "de Língua de Sinais",
  ],
  openGraph: {
    title: "AI for Brazilian Sign Language Translation - Demo Application",
    description:
    "This is the demo application for an AI model developed to translate " +
    "Brazilian Sign Language in a healthcare context. It was developed as " +
    "part of an Omdena local chapter challenge with the São Paulo chapter." +
    "For more details about the project and our achievements, please visit our Project Page website." +
    "The AI model was trained to recognize 25 different signs." +
    "Below you can test it out on some sample sign videos from our dataset, or record & upload your own!",
    url: "https://sao-paulo-brazil-chapter-brazilian-sign-language.vercel.app",
    siteName: "AI for Brazilian Sign Language Translation - Demo Application",
    images: [
    ],
    locale: "en_US",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "AI for Brazilian Sign Language Translation - Demo Application",
    description:
    "This is the demo application for an AI model developed to translate " +
    "Brazilian Sign Language in a healthcare context. It was developed as " +
    "part of an Omdena local chapter challenge with the São Paulo chapter." +
    "For more details about the project and our achievements, please visit our Project Page website." +
    "The AI model was trained to recognize 25 different signs." +
    "Below you can test it out on some sample sign videos from our dataset, or record & upload your own!",
  },
};

export default function LandingPage() {
  return <Home />;
}
