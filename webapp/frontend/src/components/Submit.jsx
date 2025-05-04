import { Button } from "@mui/material";
import { useTranslationStore } from "@/store/translationStore";
import { keyframes } from "@emotion/react";
import { getInterpretation } from "@/api/interprete";
import { toast } from "@/lib/toast";

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
    setLoading,
  } = useTranslationStore();

  const handleSubmit = async () => {
    // Call the interpretation api to 
    // interpret the sign language video
    setLoading(true);
    console.log("Submit button clicked");
    const data = label ? { label } : { video };
    try {
      const response = await getInterpretation(data);
      console.log("Response:", response);
    }
    catch (error) {
      console.log({error});
      toast.error("Error: " + error.message);
    } finally {
      setLoading(false);
    }
  };
  return (
    <Button
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
  );
};

export default SubmitButton;
