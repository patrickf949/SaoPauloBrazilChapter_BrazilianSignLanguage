import { useTranslationStore } from "@/store/translationStore";
import { Box, Typography } from "@mui/material";
import Loader, { VideoLoader } from "./Loader";
import { useEffect } from "react";

const Results = () => {
  const { loading, result } = useTranslationStore();
  useEffect(() => {
    console.log({ result });
  }, [result]);
  return (
    <Box sx={{ padding: 2, height: "100%", textAlign: "center", mt: -2 }}>
      <Typography variant="h5">Results</Typography>
      {!loading && (
        <Box
          sx={{
            height: "100%",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          {!result?.label ? (
            <Typography variant="subtitle1">
              The estimated pose landmarks and sign prediction will appear here
            </Typography>
          ) : (
            <Typography variant="subtitle1">Done!</Typography>
          )}
        </Box>
      )}
      {loading && (
        <Box
          sx={{
            height: "100%",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <Typography variant="subtitle1">Processing video... Estimating pose landmarks... Calculating features... Making a prediction...</Typography>
          <Loader />
        </Box>
      )}
      {result && (
        <Box
          sx={{
            height: "100%",
            justifyContent: "center",
            alignItems: "center",
          }}
        >
          <hr />
          {result?.videoUrl ? (
            <video
              controls
              src={result.videoUrl}
              autoPlay
              muted
              loop
              height={300}
              style={{
                borderRadius: 8,
                marginTop: "0.5rem",
                maxWidth: "100%",
                maxHeight: 250,
                justifyContent: "center",
                display: "flex",
                marginLeft: "auto",
                marginRight: "auto",
              }}
            />
          ) : (
            <div
              width="100%"
              height={300}
              style={{
                background: "#414141ff",
                borderRadius: 8,
                marginTop: "0.5rem",
                maxWidth: "100%",
                height: 300,
              }}
            />
          )}

          {loading && (
            <div
              style={{
                position: "relative",
                top: -240,
                left: 0,
                right: 0,
                bottom: 0,
                display: "flex",
                justifyContent: "center",
                alignItems: "center",
              }}
            >
              <VideoLoader />
            </div>
          )}
          <hr />
          {result?.label && (
            <Typography
              variant="h5"
              sx={{ 
                color: "success.main", 
                fontWeight: 500,
                textTransform: 'capitalize'  // This will capitalize first letter of each word
              }}
            >
              Prediction: {result?.label}
            </Typography>
          )}
        </Box>
      )}
    </Box>
  );
};

export default Results;
