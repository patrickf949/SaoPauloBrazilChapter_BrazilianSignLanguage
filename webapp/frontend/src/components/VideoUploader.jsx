import React, { useRef, useState } from 'react';
import {
    Grid,
  Box,
  Button,
  Typography,
  Input,
  Card,
  CardContent,
} from '@mui/material';

const VideoUploader = () => {
  const [videoURL, setVideoURL] = useState(null);
  const fileInputRef = useRef();

  const handleUpload = (event) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith('video/')) {
      const url = URL.createObjectURL(file);
      setVideoURL(url);
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  return (
    <Grid>
    <Card sx={{ maxWidth: 600, margin: '2rem auto', p: 2 }}>
      <CardContent>
        <Typography sx={{margin:'2rem auto '}} variant="h7" gutterBottom>
          Upload your Sign language Video for Interpretation
        </Typography>

        <Input
          type="file"
          inputRef={fileInputRef}
          onChange={handleUpload}
          sx={{ display: 'none' }}
          inputProps={{ accept: 'video/*' }}
        />

        <Button variant="contained" onClick={triggerFileSelect}>
          Choose Video File
        </Button>

        {videoURL && (
          <Box mt={3}>
            <Typography variant="subtitle1">Preview:</Typography>
            <video
              controls
              src={videoURL}
              height={300}
              style={{ 
                borderRadius: 8,
                marginTop: '0.5rem', 
                maxWidth: '100%',
                maxHeight:300,
                justifyContent: 'center',
                display: 'flex',
                marginLeft: 'auto',
                marginRight: 'auto',
            }}
            />
          </Box>
        )}
      </CardContent>
    </Card>
    </Grid>
  );
};

export default VideoUploader;
