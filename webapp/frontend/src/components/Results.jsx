import { useTranslationStore } from "@/store/translationStore";
import { Box, Typography } from "@mui/material";

const Results = () => {
  const { loading, result } = useTranslationStore();
  return (
    <Box sx={{ padding: 2, height: "100%", textAlign: "center" }}>
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
          <Typography variant="subtitle1">
            The interpretation of your selected video will appear here.
          </Typography>
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
          <Typography variant="h5">Waiting for results...</Typography>
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
          <Typography variant="h5">
            Interpretation of the selected video
          </Typography>
          <hr />
          <video
            controls
            src={result?.videoUrl}
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
          <hr />
          <Typography variant="h4">{result?.label}</Typography>
        </Box>
      )}
    </Box>
  );
};

export default Results;
