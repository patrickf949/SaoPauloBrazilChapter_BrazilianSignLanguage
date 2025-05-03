"use client";

import VideoUploader from "@/components/VideoUploader";
import SampleVideos from "@/components/SampleVideos";
import { Container, Box, Grid, Paper } from "@mui/material";

export default function Home() {
  return (
    <>
      <Container sx={{ padding: 4 }}>
        <h1>Sign Language Recognition - Brazil Sao Paulo</h1>

        <Box sx={{ flexGrow: 1, p: 2 }}>
          <Grid container spacing={2}>
            <Grid size={6}>
              <Paper elevation={3} sx={{ padding: 2 }}>
                <SampleVideos/>
                <VideoUploader />
              </Paper>
            </Grid>
            <Grid size={6}>
              <Paper elevation={3} sx={{ padding: 2 }}>
                Right Side Content
              </Paper>
            </Grid>
          </Grid>
        </Box>
      </Container>
    </>
  );
}
