# Data Directory

This directory contains all data files for the Brazilian Sign Language Recognition project. The data is version controlled using DVC.

## Directory Structure

- `raw/` - Original, immutable data
  - Video recordings of sign language
  - Any external datasets
  
- `interim/` - Intermediate data that has been transformed
  - Extracted frames
  - Preprocessed videos
  
- `processed/` - Final, canonical datasets
  - Feature vectors
  - Processed landmarks
  - Training/validation/test splits
  
- `external/` - Data from third party sources
  - Reference datasets
  - Pretrained models

## Data Version Control

This directory is tracked by DVC. To get the latest version of the data:

```bash
dvc pull
```

To add new data:
```bash
dvc add data/raw/new_data
git add data/raw/new_data.dvc
git commit -m "Add new data"
dvc push
``` 