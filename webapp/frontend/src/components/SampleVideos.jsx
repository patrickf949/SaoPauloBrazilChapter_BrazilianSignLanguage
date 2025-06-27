import { useEffect, useState, useRef } from "react";

import {
  Box,
  Card,
  CardMedia,
  CardContent,
  Typography,
  Grid,
  CircularProgress,
  IconButton,
} from "@mui/material";
import { ArrowBackIos, ArrowForwardIos, CheckBox } from "@mui/icons-material";
import { useTranslationStore } from "@/store/translationStore";

// Utility to extract Drive File ID
const extractDriveFileId = (url) => {
  const match = url.match(/\/d\/([a-zA-Z0-9_-]{10,})/);
  return match ? match[1] : null;
};

// Mock API response (simulate full links)
const fetchVideos = async () => {
  return [
    {
      url: "https://drive.google.com/file/d/1cT39EVXn-S0lOY_YrF62YC2I5OZWiJNQ/view?usp=drive_link",
      title: "Sample Video 1",
      label: "comer_uf_3.mp4",
    },
    {
      url: "https://drive.google.com/file/d/1m3wDaqv2kOai9mq_ZCHkbthTdVTNOb2x/view?usp=sharing",
      title: "Sample Video 2",
      label: "casa_vl_4.mp4",
    },
    {
      url: "https://drive.google.com/file/d/1vQGyP3820sJokH7qA4iPK5YI6bijT3tb/view?usp=drive_link",
      title: "Sample Video 3",
      label: "cabeça_sb_2.mp4",
    },
    {
      url: "https://drive.google.com/file/d/1YliBNPaGb59qb9iHbm0iT53gFuYOFper/view?usp=drive_link",
      title: "Sample Video 4",
      label: "ajudar_ne_1.mp4",
    },
  ];
};

const VideoLibrary = () => {
  const { setVideo, resetVideo, setLoadingVideo } = useTranslationStore();
  const [videos, setVideos] = useState([]);
  const [playingId, setPlayingId] = useState(null);
  const scrollRef = useRef();
  const [showLeft, setShowLeft] = useState(false);
  const [showRight, setShowRight] = useState(false);

  useEffect(() => {
    fetchVideos().then((response) => {
      const videos = response.map((video) => {
        video["id"] = extractDriveFileId(video.url);
        video["videoUrl"] = `https://drive.google.com/uc?id=${video.id}`;
        return video;
      });
      setVideos(videos);
    });
  }, []);

  useEffect(() => {
    const checkScroll = () => {
      const el = scrollRef.current;
      if (!el) return;
      setShowLeft(el.scrollLeft > 0);
      setShowRight(el.scrollLeft + el.clientWidth < el.scrollWidth);
    };

    const ref = scrollRef.current;
    if (ref) {
      ref.addEventListener("scroll", checkScroll);
      checkScroll(); // run initially
    }

    return () => {
      if (ref) ref.removeEventListener("scroll", checkScroll);
    };
  }, [videos]);

  const scrollByAmount = 300;

  const scroll = (dir) => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollBy({
      left: dir * scrollByAmount,
      behavior: "smooth",
    });
  };
  const handleVideo = (video) => {
    if (playingId === video.id) {
      resetVideo();
    } else {
      setVideo({
        videoUrl: video.id,
        video: null,
        label: video.label,
      });
    }
  };

  const handlePlay = (id) => {
    setPlayingId((prev) => (prev === id ? null : id));
  };

  return (
    <Box sx={{ p: 3, position: "relative" }}>

      {videos.length === 0 ? (
        <Box textAlign="center" py={4}>
          <CircularProgress />
        </Box>
      ) : (
        <Box sx={{ position: "relative" }}>
          {showLeft && (
            <IconButton
              onClick={() => scroll(-1)}
              sx={{
                position: "absolute",
                top: "50%",
                left: 0,
                transform: "translateY(-50%)",
                zIndex: 1,

                background: "white",
                boxShadow: 1,
                alignContent: "center",
                justifyContent: "center",
                display: "flex",
                textAlign: "center",
              }}
            >
              <ArrowBackIos
                sx={{
                  marginLeft: "auto",
                  marginRight: "auto",
                }}
              />
            </IconButton>
          )}

          {showRight && (
            <IconButton
              onClick={() => scroll(1)}
              sx={{
                position: "absolute",
                top: "50%",
                right: 0,
                transform: "translateY(-50%)",
                zIndex: 1,
                background: "white",
                boxShadow: 1,
                alignContent: "center",
                justifyContent: "center",
                display: "flex",
              }}
            >
              <ArrowForwardIos />
            </IconButton>
          )}

          <Box
            ref={scrollRef}
            sx={{
              overflowX: "auto",
              display: "flex",
              flexGrow: 1,
              "&::-webkit-scrollbar": {
                display: "none", // Chrome, Safari
              },

              gap: 2,
              scrollSnapType: "x mandatory",
              pb: 1,
              scrollBehavior: "smooth",
            }}
          >
            {videos.map((video) => {
              const id = extractDriveFileId(video.url);
              const isPlaying = playingId === id;

              return (
                <Grid
                  key={id}
                  size={{ xs: 12, sm: 6 }}
                  sx={{
                    flex: {
                      xs: "0 0 100%", // Full width on mobile
                      sm: "0 0 calc(50% - 16px)", // 2 items per row on small screens and up
                    },
                    scrollSnapAlign: "start",
                  }}
                >
                  <Card
                    onClick={() => {
                      setLoadingVideo(true);
                      handlePlay(id);
                      handleVideo(video);
                      setLoadingVideo(false);
                    }}
                    sx={{ cursor: "pointer" }}
                  >
                    <CardMedia
                      component="img"
                      sx={{
                        maxHeight: 124,
                      }}
                      image={`https://drive.google.com/thumbnail?id=${id}`}
                      alt={video.title}
                    />
                    {!isPlaying ? (
                      <>
                        <CardContent>
                          <Typography variant="subtitle1">
                            {video.title}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Click to play
                          </Typography>
                        </CardContent>
                      </>
                    ) : (
                      <Box sx={{ fontWeight: "bolder" }}>
                        <CardContent
                          sx={{
                            borderBottom: "1px solid green",
                            borderRight: "1px solid green",
                            borderLeft: "1px solid green",
                            borderRadius: "0px 0px 4px 4px",
                          }}
                        >
                          <Typography color="text.primary" variant="subtitle1">
                            <CheckBox />
                            {video.title}
                          </Typography>
                          <Typography variant="body2" color="text.secondary">
                            Click again to deselect
                          </Typography>
                        </CardContent>
                      </Box>
                    )}
                  </Card>
                </Grid>
              );
            })}
          </Box>
        </Box>
      )}
    </Box>
  );
};

export default VideoLibrary;
