import { Box, Typography, CircularProgress } from "@mui/material";
import { useTranslationStore } from "@/store/translationStore";

const VideoPlayer = () => {
  const {
    info: { videoUrl },
    loadingVideo,
  } = useTranslationStore();
  return (
    <Box 
      sx={{
        height: 280,
        width: "auto",
        border: "1px solid #e0e0e0",
        borderRadius: 2,
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
        padding: 1,
        mx: 2,
        backgroundColor: videoUrl ? "#e3f2fd" : "#f5f5f5"
      }}
    >
      {loadingVideo && (
        <Typography
          variant="subtitle1"
          sx={{ verticalAlign: "middle" }}
        >
          Loading video... <CircularProgress size={20} />
        </Typography>
      )}
      {videoUrl ? (
        <>
          <Typography variant="subtitle1" sx={{ mt: 0, mb: 1.5, px: 1.5, fontWeight: "bold" }}>
            {videoUrl.startsWith("blob") ? <b>Uploaded video:</b> : <b>Selected video:</b>}
          </Typography>
          {videoUrl.startsWith("blob") ? (
            <Box sx={{ 
              width: "100%",
              height: "calc(100% - 30px)",
              mt: -1,
              mx: -1,
              mb: -1,
              overflow: "hidden",
              borderRadius: 1
            }}>
              <video
                controls
                src={videoUrl}
                style={{
                  width: "100%",
                  height: "100%",
                  objectFit: "contain",
                }}
              />
            </Box>
          ) : (
            <Box sx={{ 
              width: "100%",
              height: "calc(100% - 30px)",
              mt: -1,
              mx: -1,
              mb: -1,
              overflow: "hidden",
              borderRadius: 1
            }}>
              <iframe
                src={`https://drive.google.com/file/d/${videoUrl}/preview`}
                allow="autoplay"
                autoPlay
                style={{
                  width: "100%",
                  height: "100%",
                  border: "none",
                }}
                allowFullScreen
              ></iframe>
            </Box>
          )}
        </>
      ) : (
        <Box sx={{ textAlign: "center" }}>
          <Typography variant="subtitle1">
          ⬆️ Browse and select a sample video ⬆️
          </Typography>
          <Typography variant="subtitle1" sx={{ my: 1 }}>
            <b>OR</b>
          </Typography>
          <Typography variant="subtitle1">
          ⬇️ Upload your own video (maximum 10 seconds) ⬇️
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default VideoPlayer;
