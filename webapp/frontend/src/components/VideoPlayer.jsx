import { Box, Typography, CircularProgress } from "@mui/material";
import { useTranslationStore } from "@/store/translationStore";

const VideoPlayer = () => {
  const {
    info: { videoUrl },
    loadingVideo,
  } = useTranslationStore();
  return (
    <Box m={3} sx={{ maxHeight: 300 }}>
      {loadingVideo && (
        <Typography
          variant="subtitle1"
          sx={{ marginTop: "0.5rem", verticalAlign: "middle" }}
        >
          Loading video... <CircularProgress />
        </Typography>
      )}
      {videoUrl ? (
        <>
          <Typography variant="subtitle1">Preview:</Typography>
          {videoUrl.startsWith("blob") ? (
            <video
              controls
              src={videoUrl}
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
            <iframe
              src={`https://drive.google.com/file/d/${videoUrl}/preview`}
              allow="autoplay"
              autoPlay
              style={{
                borderRadius: 8,
                maxWidth: "100%",
                maxHeight: 250,
                justifyContent: "center",
                display: "flex",
                marginLeft: "auto",
                marginRight: "auto",
              }}
              allowFullScreen
            ></iframe>
          )}
        </>
      ) : (
        <>
          <Typography variant="subtitle1">
            Select from our sample videos ☝️
          </Typography>
          <Typography variant="subtitle1">
            OR Upload your own video file 👇
          </Typography>
          <Typography variant="subtitle1">For interpretation</Typography>
        </>
      )}
    </Box>
  );
};

export default VideoPlayer;
