# Preprocessing Steps for Brazilian Sign Language Recognition

## Overview

This is an outline of the preprocessing pipeline. The pipeline processes videos from the four different data sources (INES, SignBank, UFV, and V-Librasil) to create two types of preprocessed outputs:
1. Landmark data for the Landmark-LSTM model
2. Video data for the CNN-LSTM model

### Data Source Characteristics
- Different video dimensions and FPS across sources, and within sources
- Varying image capture conditions
- Different signer styles and speeds

### Data Organization
- Metadata stored in `data/raw/{data_source}/metadata.csv`

### Folder Structure and File Processing
The preprocessing pipeline organizes data in the following structure:

```
/data/
|
├── raw/
│   └── combined/
│       ├── target_dataset_video_metadata.csv 
│       │   (metadata about the videos in the target dataset made using `video_scraping.ipynb`)
│       └── videos/
│           └── [original video files]
|
├── interim/
│   ├── Debug/
│   |   ├── videos/
│   |   |   └── [intermediate video files for inspecting each preprocessing step]
│   |   └── landmarks/
│   |       └── [intermediate landmark files for inspecting each preprocessing step]
│   ├── RawMotionMeasurements/
│   |   └── versionA/
│   |       └── [Raw motion measurements using version A of the method / parameters]
│   ├── RawPoseLandmarks/
│   |   └── versionA/
│   |       └── [Raw pose landmarks using version A of the method / parameters]
│   └── Videos/
│       └── [preprocessed video files before some additional processing steps]
|
└── preprocessed/
    ├── metadata_v1.csv [metadata about the videos preprocessed with version 1]
    ├── metadata_v2.csv
    ├── videos/
    │   ├── v1/
    │   │   ├── individual_metadata/
    │   │   │   └── [per-video JSON metadata that is used to make the metadata_v1.csv file]
    │   │   └── [preprocessed video files]
    │   └── v2/
    │       ├── individual_metadata/
    │       └── [preprocessed video files]
    └── landmarks/
        ├── v1/
        │   ├── individual_metadata/
        │   │   └── [per-landmark JSON metadata that is used to make the metadata_v1.csv file]
        │   └── [preprocessed landmark files]
        └── v2/
            ├── individual_metadata/
            └── [preprocessed landmark files]
```

**File Processing**:
1. **Input Files**:
   - Original videos are read from `data/raw/combined/videos/`
   - Each video is processed independently

2. **Intermediate Files**:
   - When `save_intermediate=True`, intermediate results are saved in the `debug/` subdirectories
   - These files are named with the step name as a suffix (e.g., `filename_trimmed.mp4`)
   - Useful for debugging and visualizing preprocessing steps

3. **Output Files**:
   - Preprocessed videos are saved in version-specific directories (v1/v2)
   - Each video has a corresponding JSON metadata file in the `individual_metadata/` directory
   - Full metadata is aggregated in CSV files at the root of the preprocessed directory
   - Version-specific metadata files allow tracking different preprocessing configurations

4. **Metadata Files**:
   - Individual JSON files contain detailed preprocessing information for each file
   - CSV files aggregate metadata across all files for a given version
   - Both types of metadata include original and processed file information

## Preprocessing Steps

### 1. Unify Signer Orientation
**Purpose**: Unify all signers to right-handed orientation for consistent processing.

**Process**:
- Assume all videos contain right-handed signs by default
- Record any left-handed signs in metadata
- For left-handed signs:
  - Flip video frames horizontally

**Benefits**:
- Pre-processes in subsequent steps that is based on motion measurements and pose detection will be more consistent
- We can apply data augmentation through horizontal flipping to easily include left-handed signs

### 2. Motion Detection and Trimming
**Purpose**: Remove non-informative frames before and after the actual sign performance.

**Process**:
- Apply motion detection algorithms to measure movement
- Process and combine results from multiple motion detection algorithms
- Use thresholding to determine:
  - Start frame: When significant motion begins
  - End frame: When significant motion ends
- Trim videos to the start and end of the sign performance

**Benefits**:
- Remove uninformative frames before and after the sign performance
- Focuses model on relevant sign performance

### 3. Pose Detection and Transformation
**Purpose**: Center and scale signers consistently in the frame.

**Process**:
- Detect face, pose, and hand landmarks using MediaPipe Holistic
- Use specific landmarks/measurements for:
  - Centering: Position signer in frame center
  - Scaling: Normalize signer size
- Apply transformations to:
  - Video frames
  - Landmark coordinates
- Store both original and normalized landmark data

**Benefits**:
- Consistent spatial positioning across videos means the model can focus on learning the unique features of the sign
- The positions and scales being unified means we have more control over the outcome of applying data augmentation methods
- If we reduce the dimension size, we will lose less information if the signer was scaled up (relative to the original size)

### 4. Dimension Resizing
**Purpose**: Match input requirements of respective models.

**Process**:
- Resize video frames to CNN-LSTM input dimensions
- Resize landmark coordinates to Landmark-LSTM input dimensions
- Maintain aspect ratios where possible
- Apply appropriate interpolation methods

**Benefits**:
- Consistent input dimensions for model training
- Optimized memory usage
- Preserved spatial relationships

### 5. Padding the Beginning and End of the Video
**Purpose**: Unify the real-time durations of the videos. If we had varying durations, when we sample frames, the model would have no sense of the time that has passed between frames. We have already trimmed the videos to the start and end of the sign performance, so if we pad the beginning and end equally, the sign performance will be centered in the series.

**Process**:
- Determine target sequence length
- For each video & landmark sequence:
  - Pad start and end with first and last frames respectively

**Benefits**:
- Consistent sequence lengths for model input
- Preserved temporal characteristics of signs i.e. the speed of the sign