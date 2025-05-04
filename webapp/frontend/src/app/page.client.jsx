"use client";
import dynamic from "next/dynamic"; 
const Home = dynamic(()=>import("@/containers/HomePage"),{
  ssr: false,
});

const HomePage = () => {
  return (
    <Home />
  );
}
export default HomePage;