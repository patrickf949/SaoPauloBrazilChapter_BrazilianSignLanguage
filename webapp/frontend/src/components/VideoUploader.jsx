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
    info: { videoUrl,  },
    setVideo,
    resetResult,
    resetVideo
  } = useTranslationStore();
  const fileInputRef = useRef();

  const handleUpload = (event) => {
    resetResult();
    const file = event.target.files?.[0];


    if (file && file.type.startsWith("video/")) {
      const originalName = file.name.split(".")[0]; // remove extension
      const extension = file.name.split(".").pop(); // get extension
      const sanitized = originalName.replace(/[^a-zA-Z0-9]/g, ""); // keep only alphanumerics
      const hex = Math.floor(Math.random() * 0xfffff).toString(16); // random hex
      const newFileName = `${sanitized}_${hex}.${extension}`;

      // Create a new File object with the new name
      const renamedFile = new File([file], newFileName, { type: file.type })
      const url = URL.createObjectURL(renamedFile);
      const video = document.createElement('video');
      video.src = url;

      video.onloadedmetadata = () => {
        console.log(`duration:${video.duration}`);
        if (video.duration > 10) {
          toast.error("Video must be 10 seconds or less.");
          resetVideo();
          URL.revokeObjectURL(url);
        } else {
         setVideo({ video: renamedFile, videoUrl: url, label:null });
        }
      };

      
    } else {
      toast.error("Please select a valid video file.");
    }
  };

  const triggerFileSelect = () => {
    fileInputRef.current?.click();
  };

  return (
    <Box sx={{ textAlign: 'center', mt: 2 }}>
      <Input
        type="file"
        inputRef={fileInputRef}
        onChange={handleUpload}
        sx={{ display: "none" }}
        inputProps={{ accept: "video/*" }}
      />

      <Button 
        variant="outlined" 
        onClick={triggerFileSelect}
        sx={{
          fontSize: '1.1rem',
          padding: '0.5rem 1rem',
        }}
      >
        <img
          style={{
            height: "12px",
            marginRight: "0.5rem",
          }}
          src="file.svg"
        />
        Upload Video File
      </Button>
      {videoUrl && <SubmitButton />}
    </Box>
  );
};

export default VideoUploader;
