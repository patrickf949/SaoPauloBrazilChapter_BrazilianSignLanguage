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
import sampleVideosData from '../data/sampleVideos.json';

// Utility to extract Drive File ID
const extractDriveFileId = (url) => {
  const match = url.match(/\/d\/([a-zA-Z0-9_-]{10,})/);
  return match ? match[1] : null;
};

// Mock API response (simulate full links)
const fetchVideos = async () => {
  return sampleVideosData;
};

const VideoLibrary = () => {
  const { 
    setVideo,
    resetVideo,
    setLoadingVideo,
    resetResult,
    result,
  } = useTranslationStore();
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
  }, [result]);

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
    resetResult();
    if (playingId === video.id) {
      resetVideo();
    } else {
      setVideo({
        videoUrl: video.id,
        video: null,
        label: video.label,
      });
    }
    console.log({result});
  };

  const handlePlay = (id) => {
    setPlayingId((prev) => (prev === id ? null : id));
  };

  return (
    <Box sx={{ padding: 2, height: "100%", textAlign: "center", mt: -2 }}>
      <Typography variant="h5">Submit a Video</Typography>
      {videos.length > 0 && (
        <Box
          sx={{
            height: "100%",
            display: "flex",
            justifyContent: "center",
            alignItems: "center"
          }}
        >
          <Typography variant="subtitle1">
            The video will be sent to the AI model for processing & prediction
          </Typography>
        </Box>
      )}
      <hr />
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
                height: "8px",
                width: "6px",
              },
              "&::-webkit-scrollbar-track": {
                background: "#f1f1f1",
                borderRadius: "3px",
              },
              "&::-webkit-scrollbar-thumb": {
                background: "#888",
                borderRadius: "10px",
                "&:hover": {
                  background: "#555",
                },
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
                    size={{ xs: 12, sm: 3 }}
                    sx={{
                      flex: {
                        xs: "0 0 100%", // Full width on mobile
                        sm: "0 0 calc(25% - 16px)", // 4 items per row on small screens and up
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
                                          sx={{ 
                        cursor: "pointer",
                        height:160,
                        display: 'flex',
                        flexDirection: 'column',
                        flexShrink: 1,
                      }}
                  >
                    <CardMedia
                      component="img"
                      sx={{
                        height: 100,
                        width: '100%',
                        objectFit: 'cover',
                        flexShrink: 0,
                      }}
                      image={`/thumbnails/${video.label}.jpg`}
                      alt={`${video.word_br} - ${video.word_en}`}
                    />
                    {!isPlaying ? (
                      <>
                        <CardContent sx={{ py: 0.5, pb: 0, px: 0 }}>
                          <Box sx={{ textAlign: "center" }}>
                            <Typography variant="subtitle1">
                              {video.word_br}
                            </Typography>
                            <Typography variant="body2" color="text.secondary">
                              {video.word_en}
                            </Typography>
                          </Box>
                        </CardContent>
                      </>
                    ) : (
                      <>
                        <CardContent 
                          sx={{ 
                            py: 0.5, 
                            pb: 0, 
                            px: 0,
                            backgroundColor: '#e3f2fd',
                            mt: 'auto',
                          }}
                        >
                          <Box sx={{ textAlign: "center" }}>
                            <Typography color="text.primary" variant="subtitle1" sx={{ fontWeight: "bolder" }}>
                              {video.word_br}
                            </Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ fontWeight: "bolder" }}>
                              {video.word_en}
                            </Typography>
                          </Box>
                        </CardContent>
                      </>
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
