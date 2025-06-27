import VideoUploader from "@/components/VideoUploader";
import SampleVideos from "@/components/SampleVideos";
import Results from "@/components/Results";
import { Container, Box, Grid, Paper, Typography } from "@mui/material";
import VideoPlayer from "@/components/VideoPlayer";

export default function Home() {
  return (
    <Container sx={{ padding: 4 }}>
      <Typography variant="h5" sx={{ margin: 2 }}>
        Sign Language Recognition - Brazil Sao Paulo
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
