# Project Structure

```
SaoPauloBrazilChapter_BrazilianSignLanguage/
├── data/                      # Data files
│   ├── raw/                  # Original data
│   │   ├── INES/           # INES dataset
│   │   │   └── videos/    # Video files (stored on Google Drive)
│   │   ├── SignBank/      # SignBank dataset
│   │   │   └── videos/   # Video files (stored on Google Drive)
│   │   ├── UFV/          # UFV dataset
│   │   │   └── videos/  # Video files (stored on Google Drive)
│   │   └── V-Librasil/  # V-Librasil dataset
│   │       └── videos/ # Video files (stored on Google Drive)
│   ├── interim/             # Intermediate data
│   ├── processed/           # Final datasets
│   ├── external/            # Third party sources
│   ├── papers/             # Related research papers
│   └── README.md           # Data documentation
│
├── code/                    # Source code
│   ├── data/               # Data processing
│   ├── models/             # Model implementations
│
├── notebooks/               # Jupyter notebooks
├── tests/                  # Unit tests
│   ├── data/              # Data processing tests
│   └── models/            # Model tests
│
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                # Locked dependencies
├── README.md              # Project documentation
└── STRUCTURE.md           # This file
```

## Data Management

### Video Storage
- Large video files are stored on Google Drive, not tracked in Git
- Video directories in the repository are placeholders
- Download videos to your local `videos/` directories as needed

### Data Files
- Small files like CSV files, labels, and metadata are tracked in Git
- Processed data (features, embeddings) stored in `processed/`
- Document data formats in respective directories

### Python Environment
- Managed by `uv` package manager
- Dependencies specified in `pyproject.toml`
- Versions locked in `uv.lock`

## Directory Details

### `data/`
- `raw/`: Original, immutable data
  - Dataset directories (INES, SignBank, UFV, V-Librasil)
  - Each dataset has a `videos/` subdirectory (videos on Google Drive)
  - CSV files and labels tracked in Git
- `interim/`: Intermediate processed data
- `processed/`: Final, model input datasets

### `notebooks/`
- Jupyter notebooks for exploration and development

### `tests/`
- `data/`: Data processing tests
- `models/`: Model testing

## Best Practices

1. Data Management:
   - Keep video files organized on Google Drive
   - Document video file locations and versions
   - Track small data files (CSVs, labels) in Git
   - Keep raw data immutable

2. Environment Management:
   - Use uv for dependency management
   - Keep `pyproject.toml` updated
   - Never edit `uv.lock` manually

3. Code Organization:
   - Keep notebooks for exploration
   - Write tests for critical components
   - Document data transformations

4. Documentation:
   - Document data formats and locations
   - Keep README files updated
   - Document setup steps for new team members 