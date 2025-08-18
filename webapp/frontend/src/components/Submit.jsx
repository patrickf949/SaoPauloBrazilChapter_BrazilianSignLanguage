import { Button } from "@mui/material";
import { useRef } from "react";
import { useTranslationStore } from "@/store/translationStore";
import { keyframes } from "@emotion/react";
import { useMediaQuery } from '@mui/material';
import { getInterpretation } from "@/api/interprete";
import { toast } from "@/lib/toast";
import Loader from "./Loader";

const pulse = keyframes`
  0% {
    box-shadow: 0 0 0 0 rgba(25, 118, 210, 0.7);
  }
  70% {
    box-shadow: 0 0 0 10px rgba(25, 118, 210, 0);
  }
  100% {
    box-shadow: 0 0 0 0 rgba(25, 118, 210, 0);
  }
`;
const SubmitButton = () => {
  const {
    info: { label, video },
    resetVideo,
    loading,
    setResult,
    setLoading,
    resetResult,
  } = useTranslationStore();

  const containerRef = useRef<HTMLDivElement | null>(null);
   const isMobile = useMediaQuery('(max-width:899px)');
  const scrollToBottom = () => {
    isMobile &&
    window.scrollTo({
      top: document.body.scrollHeight,
      behavior: "smooth", // smooth scroll
    });
  };


  const handleSubmit = async () => {
    // Call the interpretation api to
    // interpret the sign language video

    resetResult();
    setLoading(true);
    // Check if the
    if (!video && !label) {
      toast.error(
        "Please select a video file or select one of the sample videos."
      );
      return;
    }
    
    try {
      console.log({video});
      
      const formData = new FormData();

      if (label) {
        formData.append("label", label);
      } else {
        formData.append("file", video);
      }

      const response = await getInterpretation(formData);
      setResult({
        videoUrl: response.data.output_files.skeleton_video,
        label: response.data.prediction.label,
      });

      setLoading(false);
      scrollToBottom();
    } catch (error) {
      console.log({ error });
      toast.error(error.message);
      toast.error("Please select a video with sign language or select one of the sample videos.");
      setLoading(false);
    }
  };

  return !loading ? (
    <Button
      ref ={containerRef}
      variant="contained"
      onClick={handleSubmit}
      color="primary"
      sx={{
        margin: "1rem auto",
        animation: `${pulse} 2s 3`,
        width: "100%",
        maxHeight: 50,
      }}
    >
      Submit
    </Button>
  ) : (
    <Loader />
  );
};

export default SubmitButton;
