import { Button, Tooltip } from "@mui/material";
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

  // // Optional: scroll automatically when component mounts
  // useEffect(() => {
  //   scrollToBottom();
  // }, []);

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
      // Ensure we always pass a string to toast.error
      const errorMessage = error?.response?.data?.detail 
        ? (typeof error.response.data.detail === 'string' ? error.response.data.detail : 'An error occurred. Please try a different video.')
        : (error.message || 'An error occurred..');
      toast.error(errorMessage);
      setLoading(false);
    }
  };

  const isDisabled = loading || (!video && !label);
  
  return (
    <Tooltip 
      title={isDisabled && !loading ? "Please select a sample video or upload your own video" : ""}
      arrow
    >
      <span>
        <Button
      ref={containerRef}
      variant="contained"
      onClick={handleSubmit}
      color="primary"
      disabled={loading || (!video && !label)}
      sx={{
        margin: "1rem auto",
        animation: loading ? "none" : `${pulse} 2s 3`,
        width: "100%",
        maxHeight: 50,
        fontSize: "1rem",
      }}
    >
      {loading ? <Loader /> : "Submit"}
    </Button>
      </span>
    </Tooltip>
  );
};

export default SubmitButton;
