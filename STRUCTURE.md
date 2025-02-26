# Project Structure

```
brazilian_sign_language/
├── data/                      # Tracked by DVC
│   ├── raw/                  # Original, immutable data
│   ├── interim/              # Intermediate data
│   ├── processed/            # Final, canonical datasets
│   └── external/             # Third party sources
│
├── notebooks/
│   ├── exploration/             # Jupyter notebooks for exploration
│   ├── preprocessing/           # Data cleaning and preparation notebooks
│   └── modeling/               # Model development notebooks
│
├── src/
│   ├── data/
│   │   ├── collection/         # Scripts for downloading and scraping data
│   │   ├── cleaning/          # Data cleaning and validation
│   │   └── preprocessing/     # Feature engineering and data transformation
│   │
│   ├── models/
│   │   ├── cnn_lstm/         # CNN+LSTM model implementation
│   │   └── landmark_lstm/    # Landmark+LSTM model implementation
│   │
│   ├── features/
│   │   ├── video/           # Video feature extraction
│   │   └── landmark/        # Pose and hand landmark extraction
│   │
│   └── webapp/
│       ├── frontend/        # Web interface
│       └── backend/         # API and model serving
│
├── mlflow/                   # MLflow artifacts and metadata
│   ├── mlruns/              # Experiment tracking
│   └── models/              # Registered models
│
├── tests/                   # Unit tests
│   ├── data/
│   ├── models/
│   └── features/
│
├── configs/
│   ├── model_configs/
│   ├── preprocessing_configs/
│   └── mlflow_configs/      # MLflow configuration
│
├── models/                  # Model files (tracked by DVC)
│   ├── checkpoints/
│   └── final/
│
├── docs/                    # Documentation
│   ├── data_dictionaries/
│   ├── model_docs/
│   └── api_docs/
│
├── .venv/                  # Virtual environment (created by uv)
├── pyproject.toml          # Project metadata and dependencies
├── uv.lock                 # Lock file for dependencies
├── .dvc/                   # DVC configuration
├── .dvcignore             # DVC ignore patterns
├── dvc.yaml               # DVC pipeline definition
├── dvc.lock               # DVC pipeline lock file
├── .gitignore
├── README.md
├── STRUCTURE.md
└── LICENSE
```

## Key Components

### Data Version Control (DVC)
- `.dvc/`: DVC configuration and cache
- `dvc.yaml`: Defines data processing and training pipelines
- `dvc.lock`: Locks pipeline state
- Data and model files are tracked by DVC, not Git

### MLflow Integration
- `mlflow/`: Contains experiment tracking data
- Experiments are organized by model type
- Models can be registered and versioned
- Configurations in `configs/mlflow_configs/`

### Python Environment (uv)
- `pyproject.toml`: Project metadata and dependencies
- `uv.lock`: Locked dependencies for reproducibility
- `.venv/`: Virtual environment (automatically managed by uv)

## Directory Details

### `data/` (DVC-tracked)
- `raw/`: Original, immutable data dumps
- `interim/`: Intermediate data that has been transformed
- `processed/`: Final, canonical data sets for modeling
- `external/`: Data from third party sources

### `notebooks/`
Jupyter notebooks for exploration, analysis, and model development:
- `exploration/`: Initial data exploration and analysis
- `preprocessing/`: Data cleaning and preparation steps
- `modeling/`: Model development and evaluation

### `src/`
Source code for use in this project:
- `data/`: Scripts for data operations
- `models/`: Model implementations
- `features/`: Feature extraction code
- `webapp/`: Web application implementation

### `tests/`
Test files for different components of the project

### `configs/`
Configuration files for different components:
- Model hyperparameters
- Preprocessing parameters
- Environment configurations

### `models/`
Saved model files and checkpoints

### `docs/`
Project documentation:
- Data dictionaries
- Model architecture documentation
- API documentation
- Setup guides

### `requirements/`
Dependency management files for different environments

## Best Practices

1. Data Version Control:
   - Use DVC for tracking data and model files
   - Define clear pipelines in `dvc.yaml`
   - Keep data immutable in `raw/`

2. Experiment Tracking:
   - Log all experiments with MLflow
   - Track parameters, metrics, and artifacts
   - Register production models

3. Environment Management:
   - Use uv for dependency management
   - Keep `pyproject.toml` updated
   - Never edit `uv.lock` manually

4. Code Organization:
   - Keep notebooks for exploration
   - Use `src/` for production code
   - Write tests for critical components

5. Documentation:
   - Document data transformations
   - Track experiment configurations
   - Keep API documentation current

## Initial Setup Steps

1. Initialize uv project:
   ```bash
   uv init
   ```

2. Initialize DVC:
   ```bash
   dvc init
   dvc remote add -d storage s3://your-bucket/path  # or other remote
   ```

3. Configure MLflow:
   ```bash
   # Set up MLflow tracking URI in configs/mlflow_configs/
   ```

4. Create virtual environment:
   ```bash
   uv venv
   ``` 