import React, { useRef, useState } from "react";
import { toast } from "@/lib/toast";
import {
  Grid,
  Box,
  Button,
  Typography,
  Input,
  Card,
  CardContent,
} from "@mui/material";
import { useTranslationStore } from "@/store/translationStore";
import SubmitButton from "./Submit";

const VideoUploader = () => {
  const {
    info: { videoUrl, video },
    setVideo,
  } = useTranslationStore();
  const fileInputRef = useRef();

  const handleUpload = (event) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith("video/")) {
      const url = URL.createObjectURL(file);

      setVideo({ video: file, videoUrl: url, label:null });
    } else {
      toast.error("Please select a valid video file.");
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  return (
    <Grid>
      <Card sx={{ margin: "2rem auto", p: 2 }}>
        <CardContent sx={{}}>
          <Typography
            sx={{ margin: "2rem auto " }}
            variant="h7"
            gutterBottom
          ></Typography>

          <Input
            type="file"
            inputRef={fileInputRef}
            onChange={handleUpload}
            sx={{ display: "none" }}
            inputProps={{ accept: "video/*" }}
          />

          <Button variant="outlined" onClick={triggerFileSelect}>
            <img
              style={{
                height: "14px",
                marginRight: "0.5rem",
              }}
              height="16px"
              src="file.svg"
            />
            Upload Video File
          </Button>
          {videoUrl && <SubmitButton />}
        </CardContent>
      </Card>
    </Grid>
  );
};

export default VideoUploader;
