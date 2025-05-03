import { Button } from "@mui/material";
import { keyframes } from "@emotion/react";

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
  return (
    <Button
      variant="contained"
      color="primary"
      sx={{
        margin: "1rem auto",
        animation: `${pulse} 2s infinite`,
        width: "100%",
        maxHeight: 50,
      }}
    >
      Submit
    </Button>
  );
};

export default SubmitButton;
