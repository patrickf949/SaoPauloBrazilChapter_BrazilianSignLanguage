import { Box, Typography } from "@mui/material";
import { useTranslationStore } from "@/store/translationStore";

const VideoPlayer = () => {
    const { info: {videoUrl,video}, resetTranslation } = useTranslationStore();
  return (
    <Box mt={3} sx={{ height: 400 }}>
      {videoUrl && (
        <>
          <Typography variant="subtitle1">Preview:</Typography>
          {videoUrl.startsWith("blob") ? <video
            controls
            src={videoUrl}
            height={300}
            style={{
              borderRadius: 8,
              marginTop: "0.5rem",
              maxWidth: "100%",
              maxHeight: 300,
              justifyContent: "center",
              display: "flex",
              marginLeft: "auto",
              marginRight: "auto",
            }}
          /> :
          <iframe
            src={`https://drive.google.com/file/d/${id}/preview`}
            width="100%"
            allow="autoplay"
            autoPlay
            style={{ borderRadius: 8 }}
            allowFullScreen
          ></iframe>}
        </>
      )}
    </Box>
  );
};

export default VideoPlayer;
