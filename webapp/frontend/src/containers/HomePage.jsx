import VideoUploader from "@/components/VideoUploader";
import SampleVideos from "@/components/SampleVideos";
import Results from "@/components/Results";
import { Container, Box, Grid, Paper, Typography, Collapse } from "@mui/material";
import ExpandMore from '@mui/icons-material/ExpandMore';
import ExpandLess from '@mui/icons-material/ExpandLess';
import VideoPlayer from "@/components/VideoPlayer";
import { useState, useEffect, useRef } from "react";

export default function Home() {
  const [expanded, setExpanded] = useState(true);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setExpanded(false);
      }
    };

    const handleScroll = () => {
      setExpanded(false);
    };

    document.addEventListener('click', handleClickOutside);
    document.addEventListener('scroll', handleScroll);

    return () => {
      document.removeEventListener('click', handleClickOutside);
      document.removeEventListener('scroll', handleScroll);
    };
  }, []);

  return (
    <Container sx={{ padding: 4 }}>
      <Box ref={dropdownRef} sx={{ display: 'flex', alignItems: 'center', mb: 0, cursor: 'pointer' }} onClick={() => setExpanded(!expanded)}>
        <Typography variant="h4" sx={{ margin: 2, fontWeight: "bold" }}>
          AI for Brazilian Sign Language Translation - Demo Application
        </Typography>
        {expanded ? <ExpandLess sx={{ ml: 1 }} /> : <ExpandMore sx={{ ml: 1 }} />}
      </Box>
      
      <Collapse in={expanded}>
        <Typography variant="subtitle1" sx={{ margin: 2 }}>
          This is a demo app for an AI model developed to <b>translate Brazilian Sign Language from a healthcare context</b>. 
          <br />
          It was built during an <b>Omdena challenge with the São Paulo chapter</b>.
          For more details about the project, please visit our <a href="https://omdenaai.github.io/SaoPauloBrazilChapter_BrazilianSignLanguage/" target="_blank" rel="noopener noreferrer"><b>Project Page website</b></a>.
          <br />
          <br />
          The AI model was trained to recognize <b>25 different signs</b>.
          Below you can test it out on some sample videos from our dataset, or upload your own!
        </Typography>
      </Collapse>

      <Box sx={{ flexGrow: 1, p: 2 }}>
        <Grid 
          container
          spacing={2}
          sx={{ width: "100%", minHeight: "60vh" }}
          alignItems="stretch"
          >
          <Grid size={{ xs: 12, md: 6 }} sx={{height:'100%'}}>
            
            <Paper elevation={3} sx={{ padding: 2, height:'100%'}}>

              <SampleVideos />
              <VideoPlayer />
              <VideoUploader />
            </Paper>
          </Grid>

          <Grid size={{ xs: 12, md: 6 }} sx={{  height:'100%' }}>
            <Paper elevation={3} sx={{ padding: 2, height:'100%'}}>
              <Results />
            </Paper>
          </Grid>
        </Grid>
      </Box>
    </Container>
  );
}
