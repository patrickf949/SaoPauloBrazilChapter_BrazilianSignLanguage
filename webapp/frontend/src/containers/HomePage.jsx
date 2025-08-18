import VideoUploader from "@/components/VideoUploader";
import SampleVideos from "@/components/SampleVideos";
import Results from "@/components/Results";
import { Container, Box, Grid, Paper, Typography } from "@mui/material";
import VideoPlayer from "@/components/VideoPlayer";

export default function Home() {
  return (
    <Container sx={{ padding: 4 }}>
      <Typography variant="h4" sx={{ margin: 2, fontWeight: "bold" }}>
      AI for Brazilian Sign Language Translation - Demo Application
      </Typography>
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
