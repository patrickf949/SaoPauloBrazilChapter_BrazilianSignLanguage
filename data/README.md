# Data Directory

This directory contains data files for the Brazilian Sign Language Recognition project. Small files like .csv's with metadata can be tracked by git. Large files like videos should be stored on Google Drive.

## Directory Structure

- `raw/` - Original, immutable data
  - Video recordings of sign language
  - Any external datasets
  
- `interim/` - Intermediate data that has been transformed
  - Extracted frames
  - Preprocessed videos
  
- `processed/` - Final, model input datasets
  - Feature vectors
  - Processed landmarks
  - Training/validation/test splits