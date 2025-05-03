import { useEffect, useState,useRef } from 'react';

import {
  Box,
  Card,
  CardMedia,
  CardContent,
  Typography,
  Grid,
  CircularProgress,
  IconButton,
} from '@mui/material';
import { ArrowBackIos, ArrowForwardIos } from '@mui/icons-material';

// Utility to extract Drive File ID
const extractDriveFileId = (url) => {
  const match = url.match(/\/d\/([a-zA-Z0-9_-]{10,})/);
  return match ? match[1] : null;
};

// Mock API response (simulate full links)
const fetchVideos = async () => {
  return [
    {
      url: 'https://drive.google.com/file/d/1cT39EVXn-S0lOY_YrF62YC2I5OZWiJNQ/view?usp=drive_link',
      title: 'Sample Video 1',
    },
    {
      url: 'https://drive.google.com/file/d/1m3wDaqv2kOai9mq_ZCHkbthTdVTNOb2x/view?usp=sharing',
      title: 'Sample Video 2',
    },
    {
     url: 'https://drive.google.com/file/d/1vQGyP3820sJokH7qA4iPK5YI6bijT3tb/view?usp=drive_link',
        title: 'Sample Video 3',
    },
    {
      url: 'https://drive.google.com/file/d/1YliBNPaGb59qb9iHbm0iT53gFuYOFper/view?usp=drive_link',
      title: 'Sample Video 4',
    },
   
  ];
};


const VideoLibrary = () => {
  const [videos, setVideos] = useState([]);
  const [playingId, setPlayingId] = useState(null);
  const scrollRef = useRef();
  const [showLeft, setShowLeft] = useState(false);
  const [showRight, setShowRight] = useState(false);

  useEffect(() => {
    fetchVideos().then(setVideos);
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
      ref.addEventListener('scroll', checkScroll);
      checkScroll(); // run initially
    }

    return () => {
      if (ref) ref.removeEventListener('scroll', checkScroll);
    };
  }, [videos]);

  const scrollByAmount = 300;

  const scroll = (dir) => {
    if (!scrollRef.current) return;
    scrollRef.current.scrollBy({ left: dir * scrollByAmount, behavior: 'smooth' });
  };

  const handlePlay = (id) => {
    setPlayingId((prev) => (prev === id ? null : id));
  };

  return (
    <Box sx={{ p: 3, position: 'relative' }}>
      <Typography variant="h7" gutterBottom>
        Video Library
      </Typography>

      {videos.length === 0 ? (
        <Box textAlign="center" py={4}>
          <CircularProgress />
        </Box>
      ) : (
        <Box sx={{ position: 'relative' }}>
          {showLeft && (
            <IconButton
              onClick={() => scroll(-1)}
              sx={{
                position: 'absolute',
                top: '50%',
                left: 0,
                transform: 'translateY(-50%)',
                zIndex: 1,
                background: 'white',
                boxShadow: 1,
                alignContent: 'center',
                justifyContent: 'center',
                display: 'flex',
              }}
            >
              <ArrowBackIos />
            </IconButton>
          )}

          {showRight && (
            <IconButton
              onClick={() => scroll(1)}
              sx={{
                position: 'absolute',
                top: '50%',
                right: 0,
                transform: 'translateY(-50%)',
                zIndex: 1,
                background: 'white',
                boxShadow: 1,
                alignContent: 'center',
                justifyContent: 'center',
                display: 'flex',
              }}
            >
              <ArrowForwardIos />
            </IconButton>
          )}

          <Box
            ref={scrollRef}
            sx={{
              overflowX: 'auto',
              display: 'flex',
              gap: 2,
              scrollSnapType: 'x mandatory',
              pb: 1,
              scrollBehavior: 'smooth',
            }}
          >
            {videos.map((video) => {
              const id = extractDriveFileId(video.url);
              const isPlaying = playingId === id;

              return (
                <Grid
                  key={id}
                  sx={{
                    flex: '0 0 calc(50% - 16px)',
                    scrollSnapAlign: 'start',
                  }}
                >
                  <Card onClick={() => handlePlay(id)} sx={{ cursor: 'pointer' }}>
                    {!isPlaying ? (
                      <>
                        <CardMedia
                          component="img"
                          height="180"
                          image={`https://drive.google.com/thumbnail?id=${id}`}
                          alt={video.title}
                        />
                        <CardContent>
                          <Typography variant="subtitle1">{video.title}</Typography>
                          <Typography variant="body2" color="text.secondary">
                            Click to play
                          </Typography>
                        </CardContent>
                      </>
                    ) : (
                      <Box>
                         
                        <iframe
                            src={`https://drive.google.com/file/d/${id}/preview`}
                            width="100%"
                            allow="autoplay"
                            autoPlay
                            style={{ borderRadius: 8 }}
                            allowFullScreen
                        ></iframe>
                        <CardContent>
                          <Typography variant="subtitle1">{video.title}</Typography>
                          <Typography variant="body2" color="text.secondary">
                            Click again to close
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
