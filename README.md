# Brazilian Sign Language Recognition Project

This is a collaborative data science project under the Omdena São Paulo Chapter focused on Brazilian Sign Language (Libras) recognition. The project aims to develop machine learning models that can classify sign language videos into corresponding Portuguese words.

## Project Overview

See [STRUCTURE.md](STRUCTURE.md) for detailed project organization.

## Setup Instructions

### Prerequisites

1. Python 3.11 or higher
2. uv package manager (recommended installation via pip: `pip install uv`)

### Environment Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/OmdenaAI/SaoPauloBrazilChapter_BrazilianSignLanguage.git
   cd SaoPauloBrazilChapter_BrazilianSignLanguage
   ```

2. Install core dependencies:
   ```bash
   uv sync
   ```
   For additional dependencies:
   ```bash
   uv sync --extra <group>  # Example: uv sync --extra data
   ```
   See `pyproject.toml` for available dependency groups (data, model, app).

3. Using the environment:
   ```bash
   # Activate the environment
   uv venv activate

   # Run your code
   python your_script.py
   jupyter notebook
   ```

   Or run commands directly without activation:
   ```bash
   uv run python your_script.py
   uv run jupyter notebook
   ```

4. Adding new dependencies:
   ```bash
   uv add <package>           # Add to core dependencies
   uv add --extra data <pkg>  # Add to data processing tools
   ```

## Project Structure

```
SaoPauloBrazilChapter_BrazilianSignLanguage/
├── data/                  # Data files
│   ├── raw/              # Original data
│   │   ├── INES/        # INES dataset
│   │   │   └── videos/  # Video files (stored on Google Drive)
│   │   ├── SignBank/    # SignBank dataset
│   │   │   └── videos/  # Video files (stored on Google Drive)
│   │   ├── UFV/         # UFV dataset
│   │   │   └── videos/  # Video files (stored on Google Drive)
│   │   └── V-Librasil/  # V-Librasil dataset
│   │       └── videos/  # Video files (stored on Google Drive)
│   ├── interim/          # Intermediate processing
│   │   ├── Debug/       # Debug files for inspecting preprocessing steps
│   │   │   ├── videos/  # Intermediate video files
│   │   │   └── landmarks/ # Intermediate landmark files
│   │   ├── RawMotionMeasurements/ # Raw motion measurements
│   │   ├── RawPoseLandmarks/      # Raw pose landmarks
│   │   └── Videos/      # Preprocessed video files
│   ├── processed/        # Final datasets
│   │   ├── metadata_v*.csv # Metadata for each preprocessing version
│   │   ├── videos/      # Preprocessed videos
│   │   │   └── v*/      # Version-specific processed videos (e.g., v1, v2, v3...)
│   │   └── landmarks/   # Processed landmark data
│   │       └── v*/      # Version-specific processed landmarks (e.g., v1, v2, v3...)
│   ├── external/         # Third party data
│   └── papers/           # Related research
├── code/                 # Source code
│   ├── data/            # Data processing
│   ├── models/          # Model implementations
└── tests/               # Unit tests
```

See [STRUCTURE.md](STRUCTURE.md) for complete structure details.

## Data Management

### Video Files
- Large video files are stored on Google Drive
- Video directories in the repository structure are placeholders
- Download videos to your local `videos/` directories as needed
- Preprocessing pipeline creates versioned outputs (v1, v2, v3...) of processed videos and landmarks
- Each version has its own metadata and individual file metadata

### Data Files
- Small files like CSV files, labels, and metadata are tracked in Git
- Store processed data (features, embeddings) in `processed/`
- Document data formats in respective directories
- Metadata files track preprocessing steps and configurations